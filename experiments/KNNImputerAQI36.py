import argparse
import numpy as np
from sklearn.impute import KNNImputer

from tsl.metrics import numpy as numpy_metrics  # mesmo módulo de métricas usado no SPIN


def run_knn_imputation(
    npz_path: str,
    n_neighbors: int = 5,
    output_path: str | None = None,
):
    """
    Executa imputação via KNN a partir de um arquivo .npz do experimento SPIN
    no dataset AQI-36, tipicamente algo como:

        spin_aqi36_test_original_missing.npz

    Espera encontrar no .npz:
      - original: série completa (3D: T,N,C ou 4D: S,W,N,C)
      - missing:  série com NaNs nas posições mascaradas, mesmo shape de original
      - mask:     máscara de avaliação, mesmo shape (1 = ponto mascarado)

    Salva um novo .npz com as séries imputadas e métricas de avaliação.
    """
    # ---- Carregar dados ----
    data = np.load(npz_path)

    original = data["original"]
    missing = data["missing"]
    mask = data["mask"]

    # Campo opcional (pode não existir nesse .npz)
    missing_data_rate = data.get("missing_data_rate", None)

    # Garantir booleano
    mask_bool = mask.astype(bool)

    print(f"Carregado de {npz_path}:")
    print("  original shape:", original.shape)
    print("  missing  shape:", missing.shape)
    print("  mask     shape:", mask.shape)

    # ---- Preparar dados para o KNNImputer ----
    if original.ndim == 3:
        # Caso (T, N, C): tempo x nós x canais
        T, N, C = original.shape

        # sklearn espera (samples, features)
        # Consideramos cada instante de tempo como um sample
        # e concatenamos N*C como features.
        X_missing = missing.reshape(T, N * C)

        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="nan_euclidean",
        )

        X_imputed = imputer.fit_transform(X_missing)

        # Volta para (T, N, C)
        imputed_series = X_imputed.reshape(T, N, C)

    elif original.ndim == 4:
        # Caso (S, W, N, C): samples x janela x nós x canais (caso típico do SPIN)
        S, W, N, C = original.shape

        # Consideramos cada janela inteira (W,N,C) como um sample.
        # Achata W*N*C em features.
        X_missing = missing.reshape(S, W * N * C)

        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="nan_euclidean",
        )

        X_imputed = imputer.fit_transform(X_missing)

        # Volta para (S, W, N, C)
        imputed_series = X_imputed.reshape(S, W, N, C)

    else:
        raise ValueError(
            f"Shape de 'original' não suportado: {original.shape} "
            "(esperado 3D (T,N,C) ou 4D (S,W,N,C))."
        )

    # ---- Métricas nas posições mascaradas ----
    test_mae = numpy_metrics.mae(imputed_series, original, mask_bool)
    test_mse = numpy_metrics.mse(imputed_series, original, mask_bool)
    test_rmse = numpy_metrics.rmse(imputed_series, original, mask_bool)

    print("Resultados KNN (apenas posições mascaradas):")
    print(f"  MAE  = {test_mae:.6f}")
    print(f"  MSE  = {test_mse:.6f}")
    print(f"  RMSE = {test_rmse:.6f}")

    # ---- Montar caminho de saída (sem sobrescrever o .npz original) ----
    if output_path is None:
        # Ex.: spin_aqi36_test_original_missing.npz
        #   -> spin_aqi36_test_original_missing_knn.npz
        if npz_path.endswith(".npz"):
            output_path = npz_path.replace(".npz", "_knn.npz")
        else:
            output_path = npz_path + "_knn.npz"

    # ---- Salvar resultado ----
    np.savez(
        output_path,
        original=original,
        missing=missing,
        imputed=imputed_series,
        mask=mask,
        test_mae=test_mae,
        test_mse=test_mse,
        test_rmse=test_rmse,
        missing_data_rate=missing_data_rate,
        n_neighbors=n_neighbors,
        method="KNNImputer_window_as_sample",
    )

    print(f"Arquivo KNN salvo em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Baseline de imputação KNN para o experimento AQI-36 "
            "usando arquivos .npz do SPIN (ex.: spin_aqi36_test_original_missing.npz)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Caminho para o arquivo .npz do experimento SPIN no AQI-36 "
            "(ex.: spin_aqi36_test_original_missing.npz)"
        ),
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Número de vizinhos para o KNNImputer (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Caminho do .npz de saída. "
            "Se não for fornecido, será usado o input com sufixo '_knn' "
            "(ex.: spin_aqi36_test_original_missing_knn.npz)."
        ),
    )

    args = parser.parse_args()

    run_knn_imputation(
        npz_path=args.input,
        n_neighbors=args.neighbors,
        output_path=args.output,
    )
