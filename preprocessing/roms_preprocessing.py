from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timestamp configuration
# ---------------------------------------------------------------------------

# SPIN começa em 2005-03-01 01:00 e possui 12 amostras por dia (cada 2h)
START_DATE = dt.datetime(2005, 3, 1, 1)
SAMPLE_EVERY_HOURS = 2   # 24/12 = 2h por amostra
TIME_WINDOW = 12


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
# Enrich ROMS lat/lon (from 7x7 -> 6x6 used by SPIN)
# ---------------------------------------------------------------------------

def create_enriched_latlon(raw_dir: Path, processed_dir: Path) -> Path:
    """
    Lê roms_latlon.csv (7x7, 49 pontos) e seleciona os 36 pontos usados
    no patch 6x6 do SPIN.
    """

    base_dir = processed_dir.parent

    candidate_paths = [
        raw_dir / "roms_latlon.csv",
        processed_dir / "raw" / "roms_latlon.csv",
        processed_dir / "roms_latlon.csv",
        base_dir / "roms_latlon.csv",
        base_dir / "raw" / "roms_latlon.csv",
        base_dir.parent / "roms_latlon.csv",
        base_dir.parent / "data" / "raw" / "roms_latlon.csv",
    ]

    raw_latlon_path = None
    for p in candidate_paths:
        if p.exists():
            raw_latlon_path = p
            break

    if raw_latlon_path is None:
        tried = "\n  - ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(
            "roms_latlon.csv não encontrado. Caminhos testados:\n"
            f"  - {tried}"
        )

    print(f"[INFO] Usando roms_latlon.csv em: {raw_latlon_path}")

    df = pd.read_csv(raw_latlon_path)

    # Coordenadas alvo — ponto Gato do Mato
    gato_mato_lat = -25 - 2.34527 / 60.0
    gato_mato_lon = -42 - 59.10483 / 60.0

    df["_dist2"] = (df["latitude"] - gato_mato_lat)**2 + (df["longitude"] - gato_mato_lon)**2

    df36 = (
        df.sort_values("_dist2")
          .head(36)
          .sort_values(["latitude", "longitude"], ascending=[False, True])
          .reset_index(drop=True)
          .drop(columns=["_dist2"])
    )

    df36["point_id"] = np.arange(1, 37)
    df36["node"] = np.arange(36)

    enriched_path = processed_dir / "roms_latlon_enriched.csv"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    df36.to_csv(enriched_path, index=False)

    return enriched_path


def load_node_to_point_id(enriched_latlon_path: Path) -> Dict[int, int]:
    df = pd.read_csv(enriched_latlon_path)
    return dict(zip(df["node"].astype(int), df["point_id"].astype(int)))


# ---------------------------------------------------------------------------
# NPZ extraction
# ---------------------------------------------------------------------------

def parse_filename(npz_path: Path) -> Tuple[str, str]:
    name = npz_path.stem.lower()

    method = "spin"
    if "knn" in name:
        method = "knn"

    m = re.search(r"_(\d+)pct", name)
    if not m:
        raise ValueError(f"Não foi possível extrair o percentual: {name}")

    return method, m.group(1)


def extract_series_from_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aceita formatos:
      (T, N)
      (T, N, C)
      (T, W, N, C) -> usamos o último W e o canal C=0
    """
    data = np.load(npz_path)

    original = data["original"]
    missing = data["missing"]
    imputed = data["imputed"]

    if original.ndim == 4:
        original = original[:, -1, :, 0]
        missing = missing[:, -1, :, 0]
        imputed = imputed[:, -1, :, 0]
    elif original.ndim == 3:
        original = original[..., 0]
        missing = missing[..., 0]
        imputed = imputed[..., 0]
    elif original.ndim == 2:
        pass
    else:
        raise ValueError(
            f"Formato inesperado em {npz_path.name}: {original.shape}"
        )

    return original, missing, imputed


# ---------------------------------------------------------------------------
# Timestamp builder (2-hour resolution)
# ---------------------------------------------------------------------------

def build_timestamps(T: int) -> List[dt.datetime]:
    """
    Gera T timestamps começando em 2005-03-01 01:00, com espaçamento de 2h.
    Quando o mês de março termina, pula diretamente para 2005-06-01 01:00.
    """
    timestamps = []
    current = START_DATE

    for _ in range(T):
        timestamps.append(current)

        # avança 2 horas
        next_time = current + dt.timedelta(hours=SAMPLE_EVERY_HOURS)

        # se passou de março, pula para junho
        if next_time.month == 4:  # caiu em abril (ou após março)
            # sempre reinicia em 01/06 01:00
            next_time = dt.datetime(2005, 6, 1, 1)

        current = next_time

    return timestamps



# ---------------------------------------------------------------------------
# NPZ -> per-point CSV
# ---------------------------------------------------------------------------

def process_single_npz(
    npz_path: Path,
    processed_dir: Path,
    node_to_point: Dict[int, int],
) -> Dict[str, float]:

    method, pct = parse_filename(npz_path)

    original, missing, imputed = extract_series_from_npz(npz_path)
    T, N = original.shape

    timestamps = build_timestamps(T)

    # Cria um CSV por ponto da grade
    for node in range(N):
        point_id = node_to_point.get(node, node + 1)

        df_point = pd.DataFrame({
            "timestamp": timestamps,
            "original": original[:, node],
            "missing": missing[:, node],
            "imputed": imputed[:, node],
        })

        out_name = f"roms_{pct}pct_{method}_{point_id}.csv"
        df_point.to_csv(processed_dir / out_name, index=False)

    # Métricas agregadas
    data = np.load(npz_path)
    return {
        "file": npz_path.name,
        "method": method,
        "missing_pct": pct,
        "test_mae": float(data["test_mae"]),
        "test_mse": float(data["test_mse"]),
        "test_rmse": float(data["test_rmse"]),
    }


def process_all_npz(raw_dir: Path, processed_dir: Path, node_to_point: Dict[int, int]) -> None:
    npz_paths = sorted(raw_dir.glob("spin_gatodomato_test_*pct*.npz"))

    all_metrics = []
    for npz in npz_paths:
        print(f"[INFO] Processando {npz.name} ...")
        all_metrics.append(process_single_npz(npz, processed_dir, node_to_point))

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(processed_dir / "roms_imputation_metrics.csv", index=False)
        print(f"[INFO] Métricas salvas em {processed_dir/'roms_imputation_metrics.csv'}")
    else:
        print("[WARN] Nenhum arquivo .npz encontrado.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir, processed_dir = get_base_dirs()
    print(f"[INFO] raw_dir       = {raw_dir}")
    print(f"[INFO] processed_dir = {processed_dir}")

    enriched_path = create_enriched_latlon(raw_dir, processed_dir)
    print(f"[INFO] Arquivo enriquecido salvo em {enriched_path}")

    node_to_point = load_node_to_point_id(enriched_path)

    process_all_npz(raw_dir, processed_dir, node_to_point)


if __name__ == "__main__":
    main()
