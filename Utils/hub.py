import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from customLogger import log
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from import_utils import ENV_VARS_TRUE_VALUES
from generic import working_or_temp_dir
import huggingface_hub
from huggingface_hub import create_repo, CommitOperationAdd, create_branch, create_commit, _CACHED_NO_EXIST, ModelCard, ModelCardData, constants, get_hf_file_metadata, hf_hub_download, hf_hub_url, try_to_load_from_cache
from huggingface_hub.file_download import http_get, REGEX_COMMIT_HASH


_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint

_is_offline_mode = huggingface_hub.constants.HF_HUB_OFFLINE

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)


def is_offline_mode():
    return _is_offline_mode

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https')

def download_url(url, proxies=None):
    tmp_fd, tmp_file = tempfile.mkstemp()
    with os.fdopen(tmp_fd, 'wb') as f:
        http_get(url, f, proxies=proxies)
    return tmp_file
class PushToHubMixin:
    def _create_repo(self, repo_id : str, private : Optional[bool] = None, token : Optional[Union[bool, str]] = None, repo_url : Optional[str] = None, organization : Optional[str] = None) -> str:
        if repo_url is not None:
            warnings.warn("The repo_url argument is deprecated and will be removed in v5 of Transformers. use repo_id instead")
        
            if repo_id is not None:
                raise ValueError(
                    "repo_id and repo_url are both specified. please set only the argument repo_id" 
                )
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")

        if organization is not None:
            warnings.warn("The organization argument is deprecated and will be removed in v5 of Transformers. Set your organization directly in the repo_id passed instead (repo_id = {organization} / {model_id})")
        
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"
            
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id
    
    def _get_files_timestamps(self, working_dir : Union[str, os.PathLike]):
        return {f : os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}
    
    def _upload_modified_files(
            self, working_dir : Union[str, os.PathLike], repo_id : str, files_timestamps : Dict[str, float], commit_message : Optional[str] = None, token : Optional[Union[bool, str]] = None, create_pr : bool = False, revision : str = None, commit_description : str = None
    ):
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = 'Upload config'
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        
        modified_files = [f for f in os.listdir(working_dir) if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]]

        modified_files = [f for f in modified_files if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))]

        operations = []

        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f))
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )
        
        if revision is not None:
            create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
        
        return create_commit(repo_id=repo_id, operations=operations, commit_message=commit_message, commit_description=commit_description, token=token, create_pr=create_pr, revision=revision)
    
    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        **deprecated_kwargs,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
                Google Colab instances without any CPU OOM issues.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights in safetensors format for safer serialization.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`List[str]`, *optional*):
                List of tags to push on the Hub.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
        if repo_path_or_name is not None:
            # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
            # repo_id from the folder path, if it exists.
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead.",
                FutureWarning,
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
                )
            if os.path.isdir(repo_path_or_name):
                # repo_path: infer repo_id from the path
                repo_id = repo_id.split(os.path.sep)[-1]
                working_dir = repo_id
            else:
                # repo_name: use it as repo_id
                repo_id = repo_path_or_name
                working_dir = repo_id.split("/")[-1]
        else:
            # Repo_id is passed correctly: infer working_dir from it
            working_dir = repo_id.split("/")[-1]

        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        # Create a new empty model card and eventually tag it
        model_card = create_and_tag_model_card(
            repo_id, tags, token=token, ignore_metadata_errors=ignore_metadata_errors
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

            # Update model card if needed:
            model_card.save(os.path.join(work_dir, "README.md"))

            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )
def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None

def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if constants.HF_HUB_DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # Private arguments
    #     _raise_exceptions_for_gated_repo: if False, do not raise an exception for gated repo error but return
    #         None.
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
    if is_offline_mode() and not local_files_only:
        log("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/tree/{revision}' for available files."
                )
            else:
                return None
        return resolved_file

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    user_agent = http_user_agent(user_agent)
    try:
        # Load from URL or cache if already cached
        resolved_file = hf_hub_download(
            path_or_repo_id,
            filename,
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_gated_repo:
            return resolved_file
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to have access to it at "
            f"https://huggingface.co/{path_or_repo_id}.\n{str(e)}"
        ) from e
    except RepositoryNotFoundError as e:
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
            "having permission to this repo either by logging in with `huggingface-cli login` or by passing "
            "`token=<your_token>`"
        ) from e
    except RevisionNotFoundError as e:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
        ) from e
    except LocalEntryNotFoundError as e:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if (
            resolved_file is not None
            or not _raise_exceptions_for_missing_entries
            or not _raise_exceptions_for_connection_errors
        ):
            return resolved_file
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
            " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ) from e
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        if revision is None:
            revision = "main"
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://huggingface.co/{path_or_repo_id}/tree/{revision}' for available files."
        ) from e
    except HTTPError as err:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_connection_errors:
            return resolved_file
        raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}")
    except HFValidationError as e:
        raise EnvironmentError(
            f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a local folder or the repo_id of a model on the Hub."
        ) from e
    return resolved_file

def create_and_tag_model_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
    """
    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
    except EntryNotFoundError:
        # Otherwise create a simple model card from template
        model_description = "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
        model_card = ModelCard.from_template(card_data, model_description=model_description)

    if tags is not None:
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    return model_card