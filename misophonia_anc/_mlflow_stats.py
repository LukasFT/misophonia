import os
from collections.abc import Iterable

import pandas as pd
from mlflow.tracking import MlflowClient

from misophonia_dataset.interface import get_data_dir

MLFLOW_CACHE_DIR = get_data_dir(dataset_name="mlflow_cache")
MLFLOW_CACHE_DIR.mkdir(exist_ok=True)

assert all(k in os.environ for k in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]), (
    "MLFlow not configured"
)

RUN_INDEX = {
    "model-baseline-rep1": {
        "pretty": "Baseline",  # Median
        "mlflow": "c55334de5ae549c8a02813ed6f0f3308",
    },
    "model-baseline-rep2": {
        "pretty": "Baseline (rep. 2)",
        "mlflow": "fc19f358f21740afaa5253dcb4244002",
    },
    "model-baseline-rep3": {
        "pretty": "Baseline (rep. 3)",
        "mlflow": "b89eb23a8a1445fc95f77cd89a7cf6f1",
    },
    "model-gradclip-1": {
        "pretty": "Gradient clipping (1.0)",
        "mlflow": "8befa61b277e4b6a918dc67beb4d4385",
    },
    "model-gradclip-5": {
        "pretty": "Gradient clipping (5.0)",
        "mlflow": "176b7070fbc84e76ba0fab39eed5f7df",
    },
    "model-dropout-dec-0.05": {
        "pretty": "Dropout (dec. 0.05)",
        "mlflow": "13f54216d5664cb79b14896335aa45a7",
    },
    "model-dropout-enc-0.05": {
        "pretty": "Dropout (enc. 0.05)",
        "mlflow": "318ab952282940028e0a116396e7fcb7",
    },
    "model-dropout-dec-0.2": {
        "pretty": "Dropout (dec. 0.2)",
        "mlflow": "f363a3b52cde4d85ace50d48dfad3b2f",
    },
    "model-dropout-dec-0.5": {
        "pretty": "Dropout (dec. 0.5)",
        "mlflow": "bd6a7905116e49aa880e638dbc1b521a",
    },
    "model-lr-0.0005-wd-0.005": {
        "pretty": "Weight Decay (0.005)",
        "mlflow": "9ef2eda806134dad9be3616ee57001ab",
    },
}


def get_data_from_mlflow(run_id: str, key: str) -> pd.DataFrame:
    client = MlflowClient()
    history = client.get_metric_history(run_id, key)

    df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "step": m.step,
                m.key: m.value,
                "timestamp": m.timestamp,
            }
            for m in history
        ]
    )

    if not df.empty:
        df = df.sort_values(["step", "timestamp"])

    return df


def get_mlflow_metric_history(
    run_name: str | Iterable[str], key: str, *, exclude: tuple[str] | None = ("run_id",)
) -> pd.DataFrame:
    if not isinstance(run_name, str):
        return pd.concat([get_mlflow_metric_history(rn, key, exclude=exclude) for rn in run_name], ignore_index=True)

    run_id = RUN_INDEX.get(run_name, {}).get("mlflow")
    if run_id is None:
        raise ValueError(f"Run name {run_name} not found in index")
    file = MLFLOW_CACHE_DIR / f"{run_id}_{key.replace('/', '-')}.csv"
    if file.exists():
        df = pd.read_csv(file)
    else:
        df = get_data_from_mlflow(run_id, key)
        df["run_name"] = run_name
        df.to_csv(file, index=False)

    if exclude:
        df = df.drop(columns=list(exclude))

    col_order = ["run_name", "step", "timestamp"]
    df = df[col_order + [c for c in df.columns if c not in col_order]]
    return df


def auto_combine_mean_with_std(
    run_name: str, mean_key: str, std_key: str | None = None, *, merge_key: str = "step"
) -> pd.DataFrame:
    mean_df = get_mlflow_metric_history(run_name, mean_key)
    std_key = std_key or f"{mean_key.replace('mean', '')}_std"
    std_df = get_mlflow_metric_history(run_name, std_key)

    # Merge so all key comes from mean_df except the std_key from std_df
    df = pd.merge(mean_df, std_df[[merge_key, std_key]], on=merge_key, how="left")  # noqa: PD015
    return df
