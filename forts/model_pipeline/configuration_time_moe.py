from transformers import PretrainedConfig


class TimeMoeConfig(PretrainedConfig):
    model_type = "time_moe"

    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        attention_dropout=0.1,
        num_experts=4,
        num_experts_per_tok=2,
        use_dense=False,
        apply_aux_loss=True,
        router_aux_loss_factor=0.01,
        horizon_lengths=None,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_dense = use_dense
        self.apply_aux_loss = apply_aux_loss
        self.router_aux_loss_factor = router_aux_loss_factor
        self.horizon_lengths = horizon_lengths if horizon_lengths is not None else [1]

        super().__init__(
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
