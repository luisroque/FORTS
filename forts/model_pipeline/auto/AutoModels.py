from os import cpu_count

import torch
from neuralforecast.auto import BaseAuto
from neuralforecast.losses.pytorch import MAE
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from forts.model_pipeline.timemoe import TimeMOE


class AutoTimeMOE(BaseAuto):

    default_config = {
        "hidden_size": tune.choice([64, 128, 384]),
        "intermediate_size": tune.choice([128, 256, 512, 1536]),
        "num_hidden_layers": tune.choice([2, 4]),
        "num_attention_heads": tune.choice([2, 4, 8, 12]),
        "num_experts": tune.choice([2, 4, 8]),
        "num_experts_per_tok": tune.choice([1, 2]),
        "attention_dropout": tune.uniform(0.0, 0.2),
        "hidden_act": tune.choice(["silu", "gelu"]),
        "num_key_value_heads": tune.choice([1, 2, 4, 8]),
        "rope_theta": tune.choice([10000, 20000]),
        "max_position_embeddings": tune.choice([512, 4096]),
        "router_aux_loss_factor": tune.loguniform(1e-2, 1e-1),
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "standard"]),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=5,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoTimeMOE, self).__init__(
            cls_model=TimeMOE,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config
