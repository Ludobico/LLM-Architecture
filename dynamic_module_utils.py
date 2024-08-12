import filecmp
import importlib
import importlib.util
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

def get_relative_imports(module_file: Union[str, os.PathLike]) -> List[str]:
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Imports of the form `import .xxx`
    relative_imports = re.findall(r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from .xxx import yyy`
    relative_imports += re.findall(r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
    # Unique-ify
    return list(set(relative_imports))

def get_relative_import_files(module_file: Union[str, os.PathLike]) -> List[str]:
    no_change = False
    files_to_check = [module_file]
    all_relative_imports = []

    # Let's recurse through all relative imports
    while not no_change:
        new_imports = []
        for f in files_to_check:
            new_imports.extend(get_relative_imports(f))

        module_path = Path(module_file).parent
        new_import_files = [str(module_path / m) for m in new_imports]
        new_import_files = [f for f in new_import_files if f not in all_relative_imports]
        files_to_check = [f"{f}.py" for f in new_import_files]

        no_change = len(new_import_files) == 0
        all_relative_imports.extend(files_to_check)

    return all_relative_imports

def custom_object_save(obj : Any, folder : Union[str, os.PathLike], config : Optional[Dict] = None) ->List[str]:
    if obj.__module__ == "__main__":
        warnings.warn(
            "Saving a custom object in the __main__ module is not supported. ")
        return
    
    def _set_auto_map_in_config(_config):
        module_name = obj.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{obj.__class__.__name__}"

        if "Tokenizer" in full_name:
            slow_tokenizer_class = None
            fast_tokenizer_class = None
            if obj.__class__.__name__.endswith("Fast"):
                if getattr(obj, "slow_tokenizer_class", None) is not None:
                    slow_tokenizer = getattr(obj, 'slow_tokenizer_class')
                    slow_tok_module_name = slow_tokenizer.__module__
                    last_slow_tok_module = slow_tok_module_name.split(".")[-1]
                    slow_tokenizer_class = f"{last_slow_tok_module}.{slow_tokenizer.__name__}"
            else:
                slow_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"
            
            full_name = (slow_tokenizer_class, fast_tokenizer_class)

        if isinstance(_config, dict):
            auto_map = _config.get("auto_map", {})
            auto_map[obj._auto_class] = full_name
            _config['auto_map'] = auto_map
        elif getattr(_config, "auto_map", None) is not None:
            _config.auto_map[obj._auto_class] = full_name
        else:
            _config.auto_map = {obj._auto_class : full_name}
        
        if isinstance(config, (list, tuple)):
            for cfg in config:
                _set_auto_map_in_config(cfg)
        elif config is not None:
            _set_auto_map_in_config(config)
        
        result = []

        object_file = sys.modules[obj.__module__].__file__
        dest_file = Path(folder) / (Path(object_file).name)
        shutil.copy(object_file, dest_file)
        result.append(dest_file)

        for needed_file in get_relative_import_files(object_file):
            dest_file = Path(folder) / (Path(needed_file).name)
            shutil.copy(needed_file, dest_file)
            result.append(dest_file)
        
        return result