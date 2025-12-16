import torch
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.datasets import AirQuality
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import numpy as numpy_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

import imputation_data_prep as idp
from SPIN_model import SPINImputer

torch.set_float32_matmul_precision('medium')
    
    
def load_datamodule() -> SpatioTemporalDataModule:

    dataset = AirQuality(impute_nans=True, small=True)

    # encode time of the day and use it as exogenous variable
    covariates = {'u': dataset.datetime_encoded('day').values}

    # get adjacency matrix
    adj = dataset.get_connectivity(method="distance",
                                threshold=0.1,
                                include_self=False,
                                layout="edge_index")

    # instantiate dataset
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                        mask=dataset.training_mask,
                                        eval_mask=dataset.eval_mask,
                                        covariates=covariates,
                                        transform=MaskInput(),
                                        connectivity=adj,
                                        window=24,
                                        stride=1)

    scalers = {'target': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(val_len=0.1, test_len=0.2),
        batch_size=12,
        workers=15)
    dm.setup(stage='fit')

    return dm

    # if cfg.get('in_sample', False):
        # I think the line below makes no sense, even though the standard run of the experiment includes it.
        # dm.trainset = list(range(len(torch_dataset)))


def load_model(dataset:SpatioTemporalDataset) -> SPINImputer:

    model_kwargs = dict(n_nodes=dataset.n_nodes,
                        input_size=dataset.n_channels,
                        exog_size=dataset.input_map.u.shape[-1])

    scheduler_kwargs = dict(eta_min=0.0001,
                            T_max=300)
    
    return SPINImputer(model_kwargs=model_kwargs,
                        optim_class=torch.optim.Adam,
                        optim_kwargs=dict(lr=0.001, weight_decay=0),
                        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                        scheduler_kwargs=scheduler_kwargs,
                        whiten_prob=0.05,
                        prediction_loss_weight=1.0,
                        impute_only_missing=True,
                        warm_up_steps=0)


def load_trainer(patience=10, max_epochs=300) -> tuple[Trainer, ModelCheckpoint]:

    exp_logger = TensorBoardLogger(save_dir="data/log",
                                    name='tensorboard')

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath="data/log",
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    return Trainer(
        max_epochs=max_epochs,
        default_root_dir="data/log",
        logger=exp_logger,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=5,
        callbacks=[early_stop_callback, checkpoint_callback]), checkpoint_callback


if __name__ == "__main__":
    dm = load_datamodule()
    imputer = load_model(dm.torch_dataset)
    trainer, checkpoint_callback = load_trainer()

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    # imputer.load_model('data/log/epoch=66-step=28207.ckpt')
    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat_test, y_true_test, mask_test = (output['y_hat'], output['y'],
                            output.get('eval_mask', None))
    
    # Série original (ground truth)
    original_series_test = y_true_test

    # Série com falhas (NaN onde estava mascarado)
    missing_series_test = np.where(mask_test.astype(bool), np.nan, y_true_test)

    # Série imputada
    imputed_series_test = y_hat_test

    # Métricas nas posições mascaradas
    test_mae = numpy_metrics.mae(imputed_series_test, original_series_test, mask_test)
    test_mse = numpy_metrics.mse(imputed_series_test, original_series_test, mask_test)
    test_rmse = numpy_metrics.rmse(imputed_series_test, original_series_test, mask_test)

    # Salvar .npz de teste
    np.savez(
        "spin_aqi36_test_original_missing.npz",
        original=original_series_test,
        missing=missing_series_test,
        imputed=imputed_series_test,
        mask=mask_test,
        test_mae=test_mae,
        test_mse=test_mse,
        test_rmse=test_rmse,
    )

    res = dict(test_mae=numpy_metrics.mae(y_hat_test, y_true_test, mask_test),
                test_mre=numpy_metrics.mre(y_hat_test, y_true_test, mask_test),
                test_mape=numpy_metrics.mape(y_hat_test, y_true_test, mask_test))

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat_val, y_true_val, mask_val = (
        output["y_hat"],
        output["y"],
        output.get("eval_mask", None),
    )

    original_series_val = y_true_val
    missing_series_val = np.where(mask_val.astype(bool), np.nan, y_true_val)
    imputed_series_val = y_hat_val

    val_mae = numpy_metrics.mae(imputed_series_val, original_series_val, mask_val)
    val_mse = numpy_metrics.mse(imputed_series_val, original_series_val, mask_val)
    val_rmse = numpy_metrics.rmse(imputed_series_val, original_series_val, mask_val)

    np.savez(
        "spin_aqi36_val_original_missing.npz",
        original=original_series_val,
        missing=missing_series_val,
        imputed=imputed_series_val,
        mask=mask_val,
        val_mae=val_mae,
        val_mse=val_mse,
        val_rmse=val_rmse,
    )

    res.update(
        dict(val_mae=numpy_metrics.mae(y_hat_val, y_true_val, mask_val),
                val_rmse=numpy_metrics.rmse(y_hat_val, y_true_val, mask_val),
                val_mape=numpy_metrics.mape(y_hat_val, y_true_val, mask_val)))

    logger.info(res)