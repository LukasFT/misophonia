import functools
import json
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
    # Super models:
    "model-super-base": {
        "pretty": "Super (base)",
        "mlflow": "a969b8ffed29474da91cea6436ca317a",
        "train_size": 1_000_960,
        "train_size_per_epoch": 50_048,
    },
    "model-super-mono-channel": {
        "pretty": "Super (mono)",
        "mlflow": "431f7f510633433bb1ec66d2ecc8c3ee",
        "train_size": 1_000_960,
        "train_size_per_epoch": 50_048,
    },
    "model-super-gt-clean-mix": {
        "pretty": "Super (clean mix)",
        "mlflow": "b0677e3b9af247af85808a6d4a708171",
        "train_size": 1_000_960,
        "train_size_per_epoch": 50_048,
    },
    # Baselines:
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
    # Mono:
    "model-channels-monosplit": {
        "pretty": "Split into mono channels",
        "mlflow": "412edfa2f7db4b35ae049c3737e9edc8",
    },
    # Dimensions:
    "model-d-128": {
        "pretty": "Dim. 128",
        "mlflow": "e501b517826d44678abf18e09d1293f1",
    },
    "model-d-512": {
        "pretty": "Dim. 512",
        "mlflow": "3e482564a75446e5b51a386a0cdcff2f",
    },
    # SNR ratio
    "model-ratio-0-5": {
        "pretty": "Ratio 0-5",
        "mlflow": "f0da4c6b14774b7d9b244ff177b7a528",
    },
    "model-ratio-10-15": {
        "pretty": "Ratio 10-15",
        "mlflow": "9ac5af547ffc425bab736272a3d2d0ad",
    },
    "model-ratio-0-15": {
        "pretty": "Ratio 0-15",
        "mlflow": "37e9b4bddc244e42810b788d56ba1ee8",
    },
    # Size
    "model-size-2k": {
        "pretty": "2k training mixtures",
        "mlflow": "57bd74b49fcf4168ac252920e914afa3",
        "train_size": 2_000,
    },
    "model-size-50k": {
        "pretty": "50k training mixtures",
        "mlflow": "f3543ddd6ec14b8a849c1c83e6eb5b72",
        "train_size": 50_000,
    },
    "model-size-200k": {  # NOTE: This should be limited to the first 4 epochs, even though it ran for longer
        "pretty": "200k training mixtures",
        "mlflow": "07c8ba67420a4e3388bba31e5179a677",
        "train_size": 200_000,
    },
    "model-size-10k-of-200k": {
        "pretty": "200k training mixtures (10k per sub-epoch)",
        "mlflow": "a1ddcd5182d244069014040c171f7f97",
        "train_size": 200_000,
        "train_size_per_epoch": 10_000,
        "better_shuffle": True,
    },
    # Length
    "model-length-1sec": {
        "pretty": "1s training mixtures",
        "mlflow": "9dc0f915ec034e108dcd912efd26566e",
        "better_shuffle": True,
    },
    "model-7sec6batch": {
        "pretty": "7s training mixtures",
        "mlflow": "e908372ec2fa4daf8bbeb31e40ef3b03",
    },
    # Number of background items
    "model-backgrounds-1-3": {
        "pretty": "1-3 background items",
        "mlflow": "af303192fa4749ec80d36d9e0c347583",
    },
    # Loss
    "model-loss-with-si-snr": {
        "pretty": "10 pct. SI-SNR loss",
        "mlflow": "855c528c378145c19b6e50ba7b3f108d",
    },
    "model-loss-freq": {
        "pretty": "Frequency-domain loss",
        "mlflow": "b54ca05bf66349538cbd1a5adb755563",
    },
    # Optimization
    "model-bettershuffle": {
        "pretty": "Improved shuffling",
        "mlflow": "b265e06cce5343789e4a4e599aedda3d",
        "better_shuffle": True,
    },
    "model-lr-0.0001-wd-0.00": {
        "pretty": "Learning rate (0.0001)",
        "mlflow": "712d808508c549c080b75471d0fb03fa",
    },
    "model-lr-0.0001-wd-0.01": {
        "pretty": "Learning rate (0.0001), weight decay (0.01)",
        "mlflow": "56f61434e9624828a49e6135d28c89cb",
    },
    "model-lr-0.0005-wd-0.005": {
        "pretty": "Weight decay (0.005)",
        "mlflow": "9ef2eda806134dad9be3616ee57001ab",
    },
    "model-lr-half-on-5plateau-after-40": {
        "pretty": "Plateau scheduler",
        "mlflow": "5042d9be383040228183d1e8faea5cd6",
    },
    "model-lr-half-on-5plateau-after-40-wd-0.001": {
        "pretty": "Plateau scheduler, weight decay (0.001)",
        "mlflow": "dca5c507ff52421a88e85051180bfce4",
    },
    "model-gradclip-1": {
        "pretty": "Gradient clipping (max 1)",
        "mlflow": "8befa61b277e4b6a918dc67beb4d4385",
        "better_shuffle": True,
    },
    "model-gradclip-5": {
        "pretty": "Gradient clipping (max 5)",
        "mlflow": "176b7070fbc84e76ba0fab39eed5f7df",
        "better_shuffle": True,
    },
    "model-dropout-dec-0.05": {
        "pretty": "Dropout (dec. 0.05)",
        "mlflow": "13f54216d5664cb79b14896335aa45a7",
        "better_shuffle": True,
    },
    "model-dropout-enc-0.05": {
        "pretty": "Dropout (enc. 0.05)",
        "mlflow": "318ab952282940028e0a116396e7fcb7",
        "better_shuffle": True,
    },
    "model-dropout-dec-0.2": {
        "pretty": "Dropout (dec. 0.2)",
        "mlflow": "f363a3b52cde4d85ace50d48dfad3b2f",
        "better_shuffle": True,
    },
    "model-dropout-dec-0.5": {
        "pretty": "Dropout (dec. 0.5)",
        "mlflow": "bd6a7905116e49aa880e638dbc1b521a",
        "better_shuffle": True,
    },
    "model-ema-0.999": {
        "pretty": "EMA (decay of 0.999)",
        "mlflow": "ae6b215ad8a64fa38372a5f178b0a298",
        "better_shuffle": True,
    },
    # Control items
    "model-control-0.05": {
        "pretty": "5 pct. control items",
        "mlflow": "09a4992488cd4154b76ae069d3fb4f8b",
    },
    # Number of trigger items
    "model-foregrounds-1-2": {
        "pretty": "1-2 foreground items",
        "mlflow": "be91f4ec22c048b2a050d7ec1266d5ac",
    },
}


def get_pretty_name(run_name: str) -> str:
    return RUN_INDEX.get(run_name, {}).get("pretty", run_name)


def get_parameters_from_mlflow(run_name: str) -> dict:

    def _download_all_params(run_id: str) -> dict:
        client = MlflowClient()

        download_dir = MLFLOW_CACHE_DIR / f"{run_id}_params"
        download_dir.mkdir(exist_ok=True)

        # Get all artifacts that is parameters_xyz.json (for each key)
        artifacts = client.list_artifacts(run_id)
        param_files = [a for a in artifacts if a.path.startswith("parameters_") and a.path.endswith(".json")]
        params = {}
        for param_file in param_files:
            key = param_file.path[len("parameters_") : -len(".json")]
            local_path = download_dir / param_file.path
            client.download_artifacts(run_id, param_file.path, local_path.parent)
            with open(local_path) as f:
                params[key] = json.load(f)

        return params

    run_id = RUN_INDEX.get(run_name, {}).get("mlflow")
    if run_id is None:
        raise ValueError(f"Run name {run_name} not found in index")

    all_params_file = MLFLOW_CACHE_DIR / f"{run_id}_all_params.json"
    if all_params_file.exists():
        with open(all_params_file) as f:
            return json.load(f)
    else:
        params = _download_all_params(run_id)
        with open(all_params_file, "w") as f:
            json.dump(params, f)
        return params


def get_mlflow_metric_history(
    run_name: str | Iterable[str], key: str | Iterable[str], *, exclude: tuple[str] | None = ("run_id",)
) -> pd.DataFrame:
    if not isinstance(run_name, str):
        run_name = tuple(run_name)
        df = pd.concat([get_mlflow_metric_history(rn, key, exclude=exclude) for rn in run_name], ignore_index=True)
        # Order by run_name as given by the user
        df["run_name"] = pd.Categorical(df["run_name"], categories=run_name, ordered=True)
        df = df.sort_values(["run_name", "step", "timestamp"])
        return df

    if not isinstance(key, str):
        key = tuple(key)
        # merge on step
        keys_dfs = {k: get_mlflow_metric_history(run_name, k, exclude=exclude) for k in key}
        dfs = [keys_dfs[k][["run_name", "step", "timestamp", k]] for k in key]
        merged = functools.reduce(lambda l, r: pd.merge(l, r, on=["run_name", "step", "timestamp"], how="outer"), dfs)  # noqa: PD015
        # Add cols from first keys_dfs
        base_key = key[0]
        for col in keys_dfs[base_key].columns:
            if col not in merged.columns:
                merged[col] = keys_dfs[base_key][col]
        return merged

    def _get_data_from_mlflow(run_id: str, key: str) -> pd.DataFrame:
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

    run_id = RUN_INDEX.get(run_name, {}).get("mlflow")
    if run_id is None:
        raise ValueError(f"Run name {run_name} not found in index")
    file = MLFLOW_CACHE_DIR / f"{run_id}_{key.replace('/', '-')}.csv"
    if file.exists():
        df = pd.read_csv(file)
    else:
        df = _get_data_from_mlflow(run_id, key)
        df["run_name"] = run_name
        df.to_csv(file, index=False)

    if exclude:
        df = df.drop(columns=[c for c in exclude if c in df.columns])

    col_order = ["run_name", "step", "timestamp"]
    df = df[col_order + [c for c in df.columns if c not in col_order]]
    df["pretty_name"] = df["run_name"].apply(get_pretty_name)
    return df


def auto_combine_mean_with_std(run_name: str | Iterable[str], mean_keys: str | Iterable[str]) -> pd.DataFrame:
    mean_keys = (mean_keys,) if isinstance(mean_keys, str) else tuple(mean_keys)
    std_keys = tuple(f"{k.replace('mean', '')}_std" for k in mean_keys)
    return get_mlflow_metric_history(run_name, mean_keys + std_keys)
