from collections.abc import Mapping
from typing import Any

import torch

from tsl.engines import Imputer
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import SPINModel
from tsl.data import ImputationDataset, SpatioTemporalDataModule


torch.set_float32_matmul_precision('medium')

########################################
# imputer                              #
########################################

class SPINImputer(Imputer):
    """
    Overrides `load_model` so it works with torch > 2.6
    """
    def __init__(
        self,
        whiten_prob: float | list[float] | None = 0.05,
        prediction_loss_weight: float = 1.0,
        impute_only_missing: bool = True,
        warm_up_steps: int | tuple[int, int] = 0,
        model_kwargs: Mapping[str, Any] | None = None,
        optim_class: type | None = None,
        optim_kwargs: Mapping | None = None,
        scheduler_class = None,
        scheduler_kwargs: Mapping | None = None,
    ):

        log_metrics = {
            'mae': torch_metrics.MaskedMAE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE(),
            'mape': torch_metrics.MaskedMAPE()
        }

        if model_kwargs is not None:
            kwargs_dict:dict[str, Any] = dict(model_kwargs)
        else:
            kwargs_dict:dict[str, Any] = dict()

        SPINModel.filter_model_args_(kwargs_dict)
        kwargs_dict.update(hidden_size=32,
                            eta=3,
                            n_layers=4,
                            message_layers=1,
                            temporal_self_attention=True,
                            reweigh='softmax')

        super(SPINImputer, self).__init__(
            loss_fn=torch_metrics.MaskedMAE(), 
            scale_target=True, 
            metrics=log_metrics, 
            whiten_prob=whiten_prob, prediction_loss_weight=prediction_loss_weight,
            impute_only_missing=impute_only_missing, warm_up_steps=warm_up_steps,
            model_class=SPINModel, model_kwargs=kwargs_dict, optim_class=optim_class,
            optim_kwargs=optim_kwargs, scheduler_class=scheduler_class, scheduler_kwargs=scheduler_kwargs)
          
    def load_model(self, filename: str):
        storage = torch.load(filename, lambda storage, loc: storage, weights_only=False)
        # if predictor.model has been instantiated inside predictor
        if self.model_cls is not None:
            model_cls = storage['hyper_parameters']['model_class']
            model_kwargs = storage['hyper_parameters']['model_kwargs']
            # check model class and hyperparameters are the same
            assert model_cls == self.model_cls
            if model_kwargs is not None:
                for k, v in model_kwargs.items():
                    assert v == self.model_kwargs[k], f'{v}'
        else:
            logger.warning("Predictor with already instantiated model is "
                            f"loading a state_dict from {filename}. Cannot "
                            " check if model hyperparameters are the same.")
        self.load_state_dict(storage['state_dict'])

    
def SPIN_model_from_ckpt(ckpt_file, **kwargs):
    model_kwargs = dict(n_nodes=kwargs["n_nodes"],
                        input_size=kwargs["n_channels"],
                        exog_size=kwargs["exog_size"])

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mre': torch_metrics.MaskedMRE(),
        'mape': torch_metrics.MaskedMAPE()
    }

    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = dict(eta_min=0.0001,
                            T_max=300)

    # setup imputer
    imputer = SPINImputer(model_kwargs=model_kwargs,
                        optim_class=torch.optim.Adam,
                        optim_kwargs=dict(lr=0.001, weight_decay=0),
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        whiten_prob=0.05,
                        prediction_loss_weight=1.0,
                        impute_only_missing=True,
                        warm_up_steps=0)

    imputer.load_model(ckpt_file)

    imputer.freeze()

    return imputer

# def datamodule_from_tensor(data_tensor:torch.Tensor,
#                            valid_mask:torch.Tensor, 
#                            adjacency_matrix:torch.Tensor) -> SpatioTemporalDataModule:
    