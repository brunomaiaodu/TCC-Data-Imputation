'''
Imputation tests using the SPIN algorithm on the "gatodomato" dataset.

These scenarios are executed:
1. 30% missing data, uniformly distributed among network nodes
'''
from typing import Literal
import datetime

import torch
# from torchmetrics import Metric
import numpy as np
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule.splitters import CustomSplitter
from tsl.metrics import numpy as numpy_metrics
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy

import netCDF4
import imputation_data_prep as idp
from SPIN_model import SPINImputer

###### HYPERPARAMETERS ######
net_cdf_file_path = "/mnt/files/fabiojmo/projects/oceanml_tests/data/raw/c4ai_stress_tides_bt_2005.nc"
rng_seed = 0x7E57
min_failure_samples = 3
max_failure_samples = 24
time_window = 12
total_samples = 12*(31+28+31+30+31+30)
months_val = [2]
months_test = [3, 6]
batch_size = 6


def run_experiment(missing_data_rate=0.3,
                   missing_data_nodes: list[int] | Literal["all"] = "all",
                   total_samples: int = 7*12,
                   months_val: list[int] | None = None,
                   months_test: list[int] | None = None
                   ):
    
    torch.set_float32_matmul_precision('medium')

    ### 1) Load raw dataset  ###

    nc = netCDF4.Dataset(net_cdf_file_path)
    gato_mato_lat = -25 - 2.34527/60
    gato_mato_lon = -42 - 59.10483/60
    lat = nc['lat_rho'][:]
    lon = nc['lon_rho'][:]
    gato_mato_idx = np.argmin((lat - gato_mato_lat)**2 + (lon - gato_mato_lon)**2)
    gato_mato_eta = gato_mato_idx // lat.shape[1]
    gato_mato_xi = gato_mato_idx - (gato_mato_eta*lat.shape[1])

    ds_tensor = idp.prepare_roms_tensor(nc,
                                        use_gpu=False,
                                        end_time_idx=total_samples,
                                        start_eta=gato_mato_eta-3, end_eta=gato_mato_eta+3,
                                        start_xi=gato_mato_xi-3, end_xi=gato_mato_xi+3)

    lat_lon = {
        "lat": lat[gato_mato_eta-3:gato_mato_eta+3,gato_mato_xi-3:gato_mato_xi+3].flatten(),
        "lon": lon[gato_mato_eta-3:gato_mato_eta+3,gato_mato_xi-3:gato_mato_xi+3].flatten()
    }
    start_date = datetime.datetime(2005, 1, 1, 1)  # time of first sample (2005-01-01 01:00:00)

    ds_tensor = ds_tensor.cpu()  # The library scalers need this :-/

    print("Finished loading dataset")

    ### 2) Compute graph adjacency matrix  ###

    # ds_tensor = ds_tensor.reshape(ds_tensor.shape[0], -1, ds_tensor.shape[3])
    ds_tensor, adj = idp.gnn_style_tensor(ds_tensor)
    
    ds_tensor = ds_tensor[:,:,2].unsqueeze(2)  # selecting only zeta to test.

    ### 3) Impute NaNs --> in this case, there aren't any  ###

    ### 4) Add missing values  ###

    random = np.random.default_rng(rng_seed)
    if (missing_data_nodes == "all"):
        eval_mask = idp.missing_data_mask(ds_tensor.shape,
                                p = missing_data_rate,
                                min_sequence = min_failure_samples,
                                max_sequence = max_failure_samples,
                                rng = random)
        eval_mask = torch.tensor(eval_mask, device=ds_tensor.device)
    else:
        # select indices from nodes to impute
        ds_tensor_failing_nodes = ds_tensor[:, missing_data_nodes, :]
        failure_nodes_eval_mask = idp.missing_data_mask(ds_tensor_failing_nodes.shape,
                                p = missing_data_rate,
                                min_sequence = min_failure_samples,
                                max_sequence = max_failure_samples,
                                rng = random)

        # create the complete mask, with torch.ones everywhere else
        eval_mask = torch.ones_like(ds_tensor, device=ds_tensor.device) 
        eval_mask[:, missing_data_nodes, :] = failure_nodes_eval_mask


    ### 5) Encode covariates  ###
    
    daily_signal = {}

    daily_signal['day_sin'] = np.sin(np.arange(ds_tensor.shape[0]) * 2 * np.pi / 12)  # 12 samples per day in this dataset
    daily_signal['day_cos'] = np.cos(np.arange(ds_tensor.shape[0]) * 2 * np.pi / 12)
    covariates = {'u': pd.DataFrame(daily_signal, dtype=np.float32).values}


    ### 6) Instantiate a Dataset object  ###

    torch_dataset = ImputationDataset(target=ds_tensor,
                                        eval_mask=eval_mask,
                                        covariates=covariates,
                                        transform=MaskInput(),
                                        connectivity=adj.to_sparse(),
                                        window=time_window,
                                        stride=1)

    ### 7) Determine scaling policy  ###

    scalers = {'target': StandardScaler(axis=(0, 1))}

    ### 8) Determine splitting policy  ###

    def split_by_months(dataset, months, **kwargs):
        timestamps = np.arange(dataset.target.shape[0]).astype('timedelta64[h]') + np.datetime64(start_date)
        timestamps = timestamps.astype(datetime.datetime)
        end = time_window - 1
        start_in_months = np.isin([t.month for t in timestamps[:-end]], months)
        end_in_months = np.isin([t.month for t in timestamps[end:]], months)
        idxs_in_months = start_in_months & end_in_months
        # determine indices NOT in months (because there can be no overlap anywhere in the window)
        months_complement = np.setdiff1d(np.arange(1, 13), months)
        start_not_in_months = np.isin([t.month for t in timestamps[:-end]], months_complement)
        end_not_in_months = np.isin([t.month for t in timestamps[end:]], months_complement)
        idxs_not_in_months = start_not_in_months & end_not_in_months
        # exclude masked indices
        masked = [] if 'mask' not in kwargs.keys() else kwargs['mask']
        idxs_not_in_months_nor_masked = np.setdiff1d(np.array(np.nonzero(idxs_not_in_months)).flatten(), masked)
        idxs_in_months_andnot_masked = np.setdiff1d(np.array(np.nonzero(idxs_in_months)).flatten(), masked)

        return idxs_not_in_months_nor_masked, idxs_in_months_andnot_masked


    splitter = CustomSplitter(val_split_fn=split_by_months,
                              test_split_fn=split_by_months,
                              val_kwargs={
                                'months':months_val
                              },
                              test_kwargs={
                                  'months':months_test
                              })
    
    ### 9) Instantiate DataModule  ###

    # TODO Change the approach: write our own DataModule:
    #   implement the previous steps in the proper data preparation function for parallel processing
    #   implement spatial splitting:
    #       fix test mask
    #       fix validation mask
    #       change training mask every epoch
    #       for training/eval, the mask = mask & test_mask (it's like test nodes are always missing)
    #       for testing eval_mask = test_mask
    #       In practice, model will always impute test nodes but we won't apply backprop for their errors. (does it work?) 

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=8)
    dm.setup(stage='fit')

    # dm.trainset = list(range(len(torch_dataset)))


    ### 10) Instantiate Imputer  ###

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=torch_dataset.input_map.u.shape[-1])

    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = dict(eta_min=0.0001,
                            T_max=300)
    # setup imputer
    imputer = SPINImputer(model_kwargs=model_kwargs,
                        optim_class=torch.optim.Adam,
                        optim_kwargs=dict(lr=0.001, weight_decay=0),
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        whiten_prob=0,
                        prediction_loss_weight=1.0,
                        impute_only_missing=True,
                        warm_up_steps=0)


    ### 11) Instantiate lightning Trainer

    # logging options
    exp_logger = TensorBoardLogger(save_dir="data/log",
                                    name='tensorboard')

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=3,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath="data/log",
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=15,
        default_root_dir="data/log",
        logger=exp_logger,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=5,
        callbacks=[early_stop_callback, checkpoint_callback])

    ########################################
    # TRAIN                                #
    ########################################

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # TEST                                 #
    ########################################

    # imputer.load_model('data/log/epoch=66-step=28207.ckpt')
    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                            output.get('eval_mask', None))
    
     # Série original (ground truth)
    original_series = y_true

    # Série com falhas geradas:
    # Na convenção usual do TSL, eval_mask == 1 indica os pontos artificialmente mascarados
    missing_series = np.where(mask.astype(bool), np.nan, y_true)

    # Série imputada
    imputed_series = y_hat

    # Métricas da imputação (imputed vs original) nas posições mascaradas
    test_mae = numpy_metrics.mae(imputed_series, original_series, mask)
    test_mse = numpy_metrics.mse(imputed_series, original_series, mask)
    test_rmse = numpy_metrics.rmse(imputed_series, original_series, mask)

    pct = int(round(missing_data_rate * 100))

    # Salvar tudo em um .npz (teste)
    np.savez(
        f"spin_gatodomato_test_{pct}pct.npz",
        original=original_series,
        missing=missing_series,
        imputed=imputed_series,
        mask=mask,
        test_mae=test_mae,
        test_mse=test_mse,
        test_rmse=test_rmse,
    )

    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
                test_mre=numpy_metrics.mre(y_hat, y_true, mask),
                test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat_val, y_true_val, mask_val = (output['y_hat'], output['y'],
                            output.get('eval_mask', None))
    
    val_mae = numpy_metrics.mae(y_hat_val, y_true_val, mask_val)
    val_mse = numpy_metrics.mse(y_hat_val, y_true_val, mask_val)
    val_rmse = numpy_metrics.rmse(y_hat_val, y_true_val, mask_val)

    # Salvar as séries de validação:
    np.savez(
        f"spin_gatodomato_val_{pct}pct.npz",
        original=y_true_val,
        missing=np.where(mask_val.astype(bool), np.nan, y_true_val),
        imputed=y_hat_val,
        mask=mask_val,
        val_mae=val_mae,
        val_mse=val_mse,
        val_rmse=val_rmse,
    )

    res.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
                val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
                val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    logger.info(res)


if __name__ == "__main__":
    # run_experiment(missing_data_rate=0.25,
    #                missing_data_nodes="all",
    #                total_samples=total_samples,
    #                months_val=months_val,
    #                months_test=months_test)
    # run_experiment(missing_data_rate=0.5,
    #                missing_data_nodes="all",
    #                total_samples=total_samples,
    #                months_val=months_val,
    #                months_test=months_test)
    run_experiment(missing_data_rate=0.75,
                   missing_data_nodes="all",
                   total_samples=total_samples,
                   months_val=months_val,
                   months_test=months_test)
