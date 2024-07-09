import enum
import inspect
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"

def verify_out_features_out_indices(
        out_features : Optional[Iterable[str]], out_indices : Optional[Iterable[int]], stage_names : Optional[Iterable[str]]
):
    if stage_names is None:
        raise ValueError("Stage_names must be set for transformers backbones")
    
    if out_features is not None:
        if not isinstance(out_features, (list,)):
            raise ValueError(f"out_features must be a list got {type(out_features)}")
        if any(feat not in stage_names for feat in out_features):
            raise ValueError(f"out_features must be a subset of stage_names : {stage_names} got {out_features}")
        if len(out_features) != len(set(out_features)):
            raise ValueError(f"out features must not contain any duplicates, got {out_features}")
        if out_features != (sorted_feats := [feat for feat in stage_names if feat in out_features]):
            raise ValueError(f"out features must be in the same order as stage_names, expected {sorted_feats} got {out_features}")
    
    if out_indices is not None:
        if not isinstance(out_indices, list):
            raise ValueError(f"out indices must be a list, got {type(out_indices)}")
        
        positive_indices = tuple(idx % len(stage_names) if idx < 0 else idx for idx in out_indices)
        if any(idx for idx in positive_indices if idx not in range(len(stage_names))):
            raise ValueError(f"out indices must be valid indices for stage names {stage_names}, got {out_indices}")
        
        if len(positive_indices) != len(set(positive_indices)):
            msg = f"out indices must not contain any duplicates, got {out_indices}"
            msg += f"(equivalent to {positive_indices})" if positive_indices != out_indices else ""
            raise ValueError(msg)
        
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = [idx for _, idx in sorted(zip(positive_indices, out_indices), key=lambda x : x[0])]
            raise ValueError(f"out indices must be in the same order as stage names, expected {sorted_negative} got {out_indices}")
        
    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            raise ValueError("out features and out indices should have the same length if both are set")
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError("out features and out indices should correspond to the same stages if both are set")

def _align_output_features_output_indices(
        out_features : Optional[List[str]],
        out_indices : Optional[Union[List[int], Tuple[int]]],
        stage_names : List[str]
):
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) -1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    
    return out_features, out_indices

def get_aligned_output_features_output_indices(
        out_features : Optional[List[str]],
        out_indices : Optional[Union[List[int], Tuple[int]]],
        stage_names : List[str]
) -> Tuple[List[str], List[int]]:
    
    out_indices = list(out_indices) if out_indices is not None else None

    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    output_features, output_indices = _align_output_features_output_indices(out_features, out_indices, stage_names)

    verify_out_features_out_indices(out_features=output_features, out_indices=output_indices, stage_names=stage_names)
    return output_features, output_indices


class BackBoneMixin:
    backbone_type : Optional[BackboneType] = None

    def _init_timm_backbone(self, config) -> None:
        if getattr(self, "_backbone", None) is None:
            raise ValueError("self._backbone must be set before calling _init_timm_backbone")
        
        self.stage_names = [stage["module"] for stage in self._backbone.feature_info.info]
        self.num_features = [stage["num_chs"] for stage in self._backbone.feature_info.info]

        out_indices = list(self._backbone.feature_info.out_indices)
        out_features = self._backbone.feature_info.module_name()

        verify_out_features_out_indices(out_features, out_indices, self.stage_names)
        self._out_features, self._out_indices = out_features, out_indices
    
    def _init_transformers_backbone(self, config) -> None:
        stage_names = getattr(config, 'stage_names')
        out_features = getattr(config, 'out_features', None)
        out_indices = getattr(config, "out_indices", None)

        self.stage_names = stage_names
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features, out_indices, stage_names)
        self.num_features = None

    def _init_backbone(self, config):
        self.config = config

        self.use_timm_backbone = getattr(config, "use_timm_backbone", False)
        self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS

        if self.backbone_type == BackboneType.TIMM:
            self._init_timm_backbone(config)
        elif self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone(config)
        else:
            raise ValueError(f"backbone type {self.backbone_type} not supported")
        

    @property
    def out_features(self):
        return self._out_features
    
    @out_features.setter
    def out_features(self, out_features : List[str]):
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features, None, self.stage_names)

    @property
    def out_indices(self):
        return self._out_indices
    
    @out_indices.setter
    def out_indices(self, out_indices : Union[Tuple[int], List[int]]):
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(None, out_indices, self.stage_names)
    
    @property
    def out_feature_channels(self):
        return {stage : self.num_features[i] for i, stage in enumerate(self.stage_names)}
    
    @property
    def channels(self):
        return [self.out_feature_channels[name] for name in self.out_features]
    
    def forward_with_filtered_kwargs(self, *args, **kwargs):
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k : v for k, v in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)
    
    def forward(self, pixel_values, output_hidden_states : Optional[bool] = None, output_attentions : Optional[bool] = None, return_dict : Optional[bool] = None):
        raise NotImplementedError("This method should be implemented by the derived class.")
    
    def to_dict(self):
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output
    

class BackboneConfigMixin:
    """
    A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.
    """

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output

def load_backbone(config):
    """
    Loads the backbone model from a config object.

    If the config is from the backbone model itself, then we return a backbone model with randomly initialized
    weights.

    If the config is from the parent model of the backbone model itself, then we load the pretrained backbone weights
    if specified.
    """
    from transformers import AutoBackbone, AutoConfig

    backbone_config = getattr(config, "backbone_config", None)
    use_timm_backbone = getattr(config, "use_timm_backbone", None)
    use_pretrained_backbone = getattr(config, "use_pretrained_backbone", None)
    backbone_checkpoint = getattr(config, "backbone", None)
    backbone_kwargs = getattr(config, "backbone_kwargs", None)
    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs

    if backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

    # If there is a backbone_config and a backbone checkpoint, and use_pretrained_backbone=False then the desired
    # behaviour is ill-defined: do you want to load from the checkpoint's config or the backbone_config?
    if backbone_config is not None and backbone_checkpoint is not None and use_pretrained_backbone is not None:
        raise ValueError("Cannot specify both config.backbone_config and config.backbone")

    # If any of thhe following are set, then the config passed in is from a model which contains a backbone.
    if (
        backbone_config is None
        and use_timm_backbone is None
        and backbone_checkpoint is None
        and backbone_checkpoint is None
    ):
        return AutoBackbone.from_config(config=config, **backbone_kwargs)

    # config from the parent model that has a backbone
    if use_timm_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_timm_backbone is True")
        # Because of how timm backbones were originally added to models, we need to pass in use_pretrained_backbone
        # to determine whether to load the pretrained weights.
        backbone = AutoBackbone.from_pretrained(
            backbone_checkpoint,
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            **backbone_kwargs,
        )
    elif use_pretrained_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_pretrained_backbone is True")
        backbone = AutoBackbone.from_pretrained(backbone_checkpoint, **backbone_kwargs)
    else:
        if backbone_config is None and backbone_checkpoint is None:
            raise ValueError("Either config.backbone_config or config.backbone must be set")
        if backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(backbone_checkpoint, **backbone_kwargs)
        backbone = AutoBackbone.from_config(config=backbone_config)
    return backbone


def verify_backbone_config_arguments(
    use_timm_backbone: bool,
    use_pretrained_backbone: bool,
    backbone: Optional[str],
    backbone_config: Optional[Union[dict, "PretrainedConfig"]],
    backbone_kwargs: Optional[dict],
):
    """
    Verify that the config arguments to be passed to load_backbone are valid
    """
    if backbone_config is not None and backbone is not None:
        raise ValueError("You can't specify both `backbone` and `backbone_config`.")

    if backbone_config is not None and use_timm_backbone:
        raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

    if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")
