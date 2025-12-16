from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuração do dataset AQI-36
# ---------------------------------------------------------------------------

# Período total do AirQuality(small=True):
#   2014-05-01 00:00 até 2015-04-30 23:00 (8760 horas, amostragem de 1h)
AQI_START_DATE = dt.datetime(2014, 5, 1, 0)
AQI_TOTAL_HOURS = 24 * 365  # 8760
AQI_WINDOW_SIZE = 24        # janela usada no SPIN (window=24)
SAMPLE_EVERY_HOURS = 1      # resolução de 1 hora


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_base_dirs() -> Tuple[Path, Path]:
    """
    Returns:
        raw_dir:      <repo>/preprocessing/raw
        processed_dir:<repo>/preprocessing/processed
    """
    base_dir = Path(__file__).resolve().parent  # .../preprocessing
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    return raw_dir, processed_dir


# ---------------------------------------------------------------------------
# Carrega lat/lon do AQI-36 e monta mapeamento node -> point_id
# ---------------------------------------------------------------------------

def create_aqi_latlon_enriched(raw_dir: Path, processed_dir: Path) -> Path:
    """
    Lê aqi36_latlon.csv e garante um formato canônico:

        point_id,station_id(optional),latitude,longitude,node

    onde:
        point_id = 1..36
        node     = 0..35 (índices usados pelo modelo)

    O arquivo de entrada pode ser:
        - point_id,latitude,longitude
        - OU station_id,latitude,longitude  (convertemos para point_id = 1..N)
    """

    base_dir = processed_dir.parent

    candidate_paths = [
        raw_dir / "aqi36_latlon.csv",
        processed_dir / "raw" / "aqi36_latlon.csv",
        processed_dir / "aqi36_latlon.csv",
        base_dir / "aqi36_latlon.csv",
        base_dir / "raw" / "aqi36_latlon.csv",
        base_dir.parent / "aqi36_latlon.csv",
        base_dir.parent / "data" / "raw" / "aqi36_latlon.csv",
    ]

    latlon_path = None
    for p in candidate_paths:
        if p.exists():
            latlon_path = p
            break

    if latlon_path is None:
        tried = "\n  - ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(
            "aqi36_latlon.csv não encontrado. Caminhos testados:\n"
            f"  - {tried}"
        )

    print(f"[INFO] Usando aqi36_latlon.csv em: {latlon_path}")

    df = pd.read_csv(latlon_path)

    # Normaliza colunas
    if "point_id" in df.columns:
        # Já vem enumerado 1..N
        df = df[["point_id", "latitude", "longitude"]].copy()
        df = df.sort_values("point_id").reset_index(drop=True)
    elif "station_id" in df.columns:
        # Converte station_id -> point_id 1..N
        df = df[["station_id", "latitude", "longitude"]].copy()
        df = df.sort_values("station_id").reset_index(drop=True)
        df["point_id"] = np.arange(1, len(df) + 1, dtype=int)
    else:
        raise ValueError(
            "aqi36_latlon.csv deve conter 'point_id' ou 'station_id' "
            f"(colunas atuais: {list(df.columns)})"
        )

    if len(df) != 36:
        print(f"[WARN] Esperado 36 estações, encontrado {len(df)}.")

    # Garante ordenação por point_id
    df = df.sort_values("point_id").reset_index(drop=True)

    # Cria coluna node (0..N-1)
    df["node"] = df["point_id"] - 1

    enriched_path = processed_dir / "aqi36_latlon_enriched.csv"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(enriched_path, index=False)

    print(f"[INFO] Arquivo enriquecido salvo em {enriched_path}")
    return enriched_path


def load_node_to_point_id(enriched_latlon_path: Path) -> Dict[int, int]:
    df = pd.read_csv(enriched_latlon_path)
    return dict(zip(df["node"].astype(int), df["point_id"].astype(int)))


# ---------------------------------------------------------------------------
# NPZ extraction
# ---------------------------------------------------------------------------

def detect_method(npz_path: Path) -> str:
    """
    Define o método a partir do nome do arquivo:
      - contém 'knn'  -> 'knn'
      - caso contrário -> 'spin'
    """
    name = npz_path.stem.lower()
    if "knn" in name:
        return "knn"
    return "spin"


def extract_series_from_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Aceita formatos:
      (T, N)
      (T, N, C)
      (S, W, N, C) -> usamos o último W (W-ésimo passo) e canal 0

    Retorna:
        original_2d, missing_2d, imputed_2d com shape (S, N)
        window_size (W) usado no arquivo (p/ reconstruir timestamps)
    """
    data = np.load(npz_path)

    original = data["original"]
    missing = data["missing"]
    imputed = data["imputed"]

    if original.ndim == 4:
        # (S, W, N, C)
        S, W, N, C = original.shape
        original_2d = original[:, -1, :, 0]
        missing_2d = missing[:, -1, :, 0]
        imputed_2d = imputed[:, -1, :, 0]
        window_size = W
    elif original.ndim == 3:
        # (T, N, C)
        T, N, C = original.shape
        original_2d = original[..., 0]
        missing_2d = missing[..., 0]
        imputed_2d = imputed[..., 0]
        window_size = 1  # aqui não há janela explícita
    elif original.ndim == 2:
        original_2d = original
        missing_2d = missing
        imputed_2d = imputed
        window_size = 1
    else:
        raise ValueError(
            f"Formato inesperado para 'original' em {npz_path.name}: {original.shape}"
        )

    if original_2d.shape != missing_2d.shape or original_2d.shape != imputed_2d.shape:
        raise ValueError(
            f"Shapes incompatíveis em {npz_path.name}: "
            f"original={original_2d.shape}, missing={missing_2d.shape}, imputed={imputed_2d.shape}"
        )

    return original_2d, missing_2d, imputed_2d, window_size


# ---------------------------------------------------------------------------
# Reconstrução de timestamps para o TESTE do AQI-36
# ---------------------------------------------------------------------------

def build_test_timestamps(
    n_windows: int,
    window_size: int,
    start_date: dt.datetime = AQI_START_DATE,
    total_hours: int = AQI_TOTAL_HOURS,
    step_hours: int = SAMPLE_EVERY_HOURS,
) -> List[dt.datetime]:
    """
    Reconstrói os timestamps (um por janela, tomando o último passo da janela)
    para o *conjunto de teste* do AQI-36.

    Lógica:
      - Nº total de janelas possíveis no ano: total_windows = total_hours - window_size + 1
      - Sabendo n_windows (tamanho do teste), assumimos que o teste ocupa o final da série.
        Então:
            first_window_idx = total_windows - n_windows
      - O último passo da janela 0 está em:
            last_idx_0 = first_window_idx + window_size - 1
      - O timestamp da janela i é:
            t_i = start_date + (last_idx_0 + i) * 1h
    """
    total_windows = total_hours - window_size + 1
    first_window_idx = total_windows - n_windows     # índice (em horas) do INÍCIO da primeira janela de teste
    last_idx_0 = first_window_idx + window_size - 1 # índice (em horas) do último passo da primeira janela

    timestamps = [
        start_date + dt.timedelta(hours=last_idx_0 + i * step_hours)
        for i in range(n_windows)
    ]
    return timestamps


# ---------------------------------------------------------------------------
# NPZ -> per-station CSV + metrics CSV
# ---------------------------------------------------------------------------

def process_single_npz(
    npz_path: Path,
    processed_dir: Path,
    node_to_point: Dict[int, int],
) -> Dict[str, float]:
    """
    Converte um arquivo .npz (teste AQI-36) em vários CSVs, um por estação:

        aqi36_{method}_{point_id}.csv

    com colunas:
        timestamp,original,missing,imputed

    Também devolve as métricas agregadas contidas no .npz, além do
    percentual de pontos mascarados (missing_pct).
    """
    method = detect_method(npz_path)

    original, missing, imputed, window_size = extract_series_from_npz(npz_path)
    T, N = original.shape

    # Reconstrói timestamps do TESTE
    timestamps = build_test_timestamps(
        n_windows=T,
        window_size=window_size,
    )

    # Um CSV por estação
    for node in range(N):
        point_id = node_to_point.get(node, node + 1)

        df_point = pd.DataFrame(
            {
                "timestamp": timestamps,
                "original": original[:, node],
                "missing": missing[:, node],
                "imputed": imputed[:, node],
            }
        )

        out_name = f"aqi36_{method}_{point_id}.csv"
        out_path = processed_dir / out_name
        df_point.to_csv(out_path, index=False)

    # Métricas agregadas e percentual de missing
    data = np.load(npz_path)
    mask = data["mask"].astype(bool)

    missing_pct = 100.0 * float(mask.sum()) / mask.size

    # Usa métricas já salvas no .npz
    test_mae = float(data["test_mae"])
    test_mse = float(data["test_mse"])
    test_rmse = float(data["test_rmse"])

    metrics = {
        "file": npz_path.name,
        "method": method,
        "missing_pct": missing_pct,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
    }
    return metrics


def process_all_npz(raw_dir: Path, processed_dir: Path, node_to_point: Dict[int, int]) -> None:
    """
    Percorre todos os arquivos *aqi36*test_original_missing*.npz em raw_dir,
    gera os CSVs por estação em processed_dir e um CSV único com métricas.
    """
    npz_paths = sorted(raw_dir.glob("*aqi36*test_original_missing*.npz"))

    if not npz_paths:
        print("[WARN] Nenhum arquivo *aqi36*test_original_missing*.npz encontrado.")
        return

    all_metrics = []
    for npz_path in npz_paths:
        print(f"[INFO] Processando {npz_path.name} ...")
        metrics = process_single_npz(npz_path, processed_dir, node_to_point)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_out = processed_dir / "aqi36_imputation_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"[INFO] Métricas salvas em {metrics_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir, processed_dir = get_base_dirs()
    print(f"[INFO] raw_dir       = {raw_dir}")
    print(f"[INFO] processed_dir = {processed_dir}")

    enriched_path = create_aqi_latlon_enriched(raw_dir, processed_dir)
    node_to_point = load_node_to_point_id(enriched_path)

    process_all_npz(raw_dir, processed_dir, node_to_point)


if __name__ == "__main__":
    main()
