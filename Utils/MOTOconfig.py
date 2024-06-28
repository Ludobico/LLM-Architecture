import torch

class ModelConfig:
    torch.cuda.empty_cache()
    def __init__(self, config):
        self.config = config
    @property
    def get_model_architecture(self):
        print("model_type : {0}".format(self.config.config.architectures))
        print("num_hidden_layers : {0}".format(self.config.config.num_hidden_layers))
        print("num_attention_heads : {0}".format(self.config.config.num_attention_heads))
        print("hidden_size : {0}".format(self.config.config.hidden_size))
        print("intermediate_size : {0}".format(self.config.config.intermediate_size))
        print("num_key_value_heads : {0}".format(self.config.config.num_key_value_heads))
    
    @property
    def get_token_settings(self):
        print("vocab_size : {0}".format(self.config.config.vocab_size))
        print("max_position_embeddings : {0}".format(self.config.config.max_position_embeddings))
        print("bos_token_id : {0}".format(self.config.config.bos_token_id))
        print("eos_token_id : {0}".format(self.config.config.eos_token_id))
    
    @property
    def get_initialization_settings(self):
        print("initializer_range : {0}".format(self.config.config.initializer_range))
        print("pretraining_tp : {0}".format(self.config.config.pretraining_tp))
    
    @property
    def get_activation_setting(self):
        print("hidden_activation : {0}".format(self.config.config.hidden_act))
    
    @property
    def get_normalization_setting(self):
        print("rms_norm_eps : {0}".format(self.config.config.rms_norm_eps))
    
    @property
    def get_rope_settings(self):
        print("rope_scaling : {0}".format(self.config.config.rope_scaling))
        print("rope_theta : {0}".format(self.config.config.rope_theta))
