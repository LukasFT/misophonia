"""A collection of useful helper functions"""

# ruff: noqa: ANN001 # TODO: Improve quality

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eliot
import numpy as np
import pydantic
import torch
import torch.nn.functional as F  # noqa: N812  # noqa: N812
import webdataset as wds
import yaml
from scipy import signal
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from misophonia_dataset.interface import BaseModel, MisophoniaItem, SplitT
from misophonia_dataset.main import get_default_datasets_names
from misophonia_dataset.misophonia_dataset import MisophoniaDatasetSplit

# Initialize random generator for reproducibility
rng = np.random.default_rng()


##############
# Prprocess Utils #
##############


def preprocess_to_webdataset_pt(
    shards_dir: str | Path,
    dataset_split: MisophoniaDatasetSplit,
    *,
    samples_per_shard: int,
    num_workers: int = None,
    show_progress: bool = True,
) -> str:
    """
    Preprocess GeneratedMisophoniaDataset into WebDataset .tar shards using multithreading.
    Saves tensors as mix.pt, gt.pt, and label.pt.

    Args:
        shards_dir: Directory to save the .tar shards. Assumes shards_dir already contains split in name
        dataset_split: The dataset split to preprocess.
        samples_per_shard: Number of samples per .tar shard
        num_workers: Number of threads to use for parallel processing. If None, defaults to number of CPU cores.

    Returns:
        A glob pattern for the generated .tar shards. Used for loading wds.WebDataset.
    """

    def process_item(idx) -> dict:
        item = dataset_split[idx]
        # This function should call your actual preprocessing
        # preprocess_item_to_arrays -> returns (X, y, label_vec)
        mix_array, gt_array, label_array = preprocess_item_to_tensors(item)

        sample = {"__key__": f"{idx:09d}", "mix.npy": mix_array, "label.npy": label_array, "gt.npy": gt_array}
        return sample

    def preprocess_item_to_tensors(item: MisophoniaItem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetches the audio in the form of np.ndarray for the binaural mix and ground truth.
        Also fetches the label vector as np.ndarray.
        """
        # Example preprocessing (replace with actual logic)
        mix = item.get_mix_audio()
        gt = item.get_ground_truth_audio()
        label_vec = item.label_vector

        return (mix, label_vec, gt)

    num_workers = num_workers or os.cpu_count() or 1

    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(shards_dir / "data-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            size = len(dataset_split)
            results = executor.map(process_item, range(size))
            if show_progress:
                results = tqdm(results, total=size, desc=f"Saving {dataset_split.split} items")

            for result in results:
                sink.write(result)

    shard_glob = str(shards_dir / "data-*.tar")
    return shard_glob


class MisophoniaDatasetPreprocessedConfig(BaseModel):
    from_premade: str | None = pydantic.Field(
        None,
        description="Whether to use a premade dataset with the given name. If a non-empty string, use that name of the premade dataset.",
    )
    generated_source_data: list[str] = pydantic.Field(
        get_default_datasets_names(),
        description="If from_premade is False, will generate dataset using the given source datasets. See GeneratedMisophoniaDataset for options.",
    )
    generated_config: dict | None = pydantic.Field(
        None,
        description="If premade_config not given, will generate dataset using the given config. See GeneratedMisophoniaDataset.get_split for options.",
    )


class MisophoniaANCConfig(BaseModel):
    dataset_splits: dict[SplitT, MisophoniaDatasetPreprocessedConfig] = pydantic.Field(
        ..., description="For each split, the config for the preprocessed dataset to use for training/eval."
    )

    num_epochs: int = pydantic.Field(10, description="Number of epochs to train for.")
    batch_size: int = pydantic.Field(1, description="Batch size for training.")

    model_params: dict = pydantic.Field(
        {}, description="Dictionary of parameters to initialize the model. See the MisophoniaANCNet class for options."
    )

    mlflow_experiment: str | None = pydantic.Field(
        None, description="MLflow experiment name to log training metrics to."
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MisophoniaANCConfig":
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Cannot load config since file does not exist: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def custom_collate_fn(
    batch: list[list[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # Pad the audio to all be the same length (the length of the longest audio in the batch)
    max_len = max([mix.shape[-1] for mix, _, _ in batch])

    mixes = []
    gts = []
    labels = []
    masks = []
    for mix, label, gt in batch:
        pad_len = max_len - mix.shape[-1]
        assert pad_len >= 0, "Error calculating batch padding"

        mix = F.pad(torch.from_numpy(mix).to(torch.float32), (0, pad_len))  # Convert and pad mix
        gt = F.pad(torch.from_numpy(gt).to(torch.float32), (0, pad_len))  # Convert and pad gt

        mask = torch.zeros_like(mix)
        mask[:, -pad_len:] = 1.0

        mixes.append(mix)
        gts.append(gt)
        labels.append(torch.from_numpy(label).to(torch.float32))  # Convert label
        masks.append(mask)

    inputs = {
        "mix": torch.stack(mixes),
        "label_vector": torch.stack(labels),
    }
    gt = torch.stack(gts)
    masks = torch.stack(masks)

    return inputs, gt, masks


def make_dataloader(files: list[str | Path], *, batch_size: int, num_workers: int) -> wds.WebLoader:
    assert len(files) > 0
    eliot.log_message(f"Loading data from `{files[0]}` etc...", level="debug")
    eliot.log_message(
        f"Using {num_workers} workers loading WebDataset (total CPU count = {os.cpu_count()}, allocated = {get_allocated_cpus()}).",
        level="debug",
    )
    data = (
        wds.WebDataset(
            files,
            empty_check=False,
            shardshuffle=1,  # Number of shards to keep in memory at the time (as I understand it)
        )
        .shuffle(batch_size)  # Number of samples to shuffle in memory at the time (as I understand it)
        .decode("torch")  # converts the saved numpy arrays to tensors
        .to_tuple("mix.npy", "gt.npy", "label.npy")
        .batched(
            batch_size,
            collation_fn=custom_collate_fn,  # Make batches of the same size
        )
    )

    return wds.WebLoader(
        data,
        batch_size=None,  # We set batch size in the WebDataset pipeline, so we set it to None here
        num_workers=num_workers,
    )


def get_allocated_cpus() -> int:
    """Returns the number of CPUs allocated to the current process, accounting for environments like SLURM and Docker."""

    # 1. Try SLURM environment variable
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)

    # 2. Try scheduler affinity (works for Docker/K8s/SLURM with cgroups)
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    # 3. Fallback to total system CPUs
    return os.cpu_count() or 1


def mod_pad(x, chunk_size, pad):  # noqa: ANN202  # TODO
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


def model_size(model) -> float:
    """
    Returns size of the `model` in millions of parameters.
    """
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_train_params / 1e6


def run_time(model, inputs, *, profiling: bool = False) -> float:
    """
    Returns runtime of a model in ms.
    """
    # Warmup
    for _ in range(100):
        output = model(*inputs)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(*inputs)  # noqa: F841

    # Print profiling results
    if profiling:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # Return runtime in ms
    return prof.profiler.self_cpu_time_total / 1000


##############
# Eval Utils #
##############


def compute_itd(s_left, s_right, sr, t_max=None) -> float:
    corr = signal.correlate(s_left, s_right)
    corr /= np.max(corr)

    mid = len(corr) // 2 + 1

    cc = np.concatenate((corr[-mid:], corr[:mid]))

    if t_max is not None:
        cc = np.concatenate([cc[-t_max + 1 :], cc[: t_max + 1]])
    else:
        t_max = mid

    tau = np.argmax(np.abs(cc))
    tau -= t_max

    return tau / sr * 1e6


def compute_ild(s_left, s_right) -> np.ndarray:
    sum_sq_left = np.sum(s_left**2, axis=-1)
    sum_sq_right = np.sum(s_right**2, axis=-1)
    return 10 * np.log10(sum_sq_left / sum_sq_right)


def itd_diff(s_est, s_gt, sr) -> np.ndarray:
    """
    Computes the ITD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    tmax = int(round(1e-3 * sr))
    itd_est = compute_itd(s_est[..., 0, :], s_est[..., 1, :], sr, tmax)
    itd_gt = compute_itd(s_gt[..., 0, :], s_gt[..., 1, :], sr, tmax)
    return np.abs(itd_est - itd_gt)


def ild_diff(s_est, s_gt) -> np.ndarray:
    """
    Computes the ILD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
    ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])
    return np.abs(ild_est - ild_gt)


# TODO: Remove unused (commented out) functions

# import numpy.fft as fft  # Semantic Hearinc also used mklfft, which is optimized for Intel
# import pyroomacoustics as pra  # Not installed, do we need it?
# import torch
# from scipy.fft import irfft, rfft


# def tdoa2(x1, x2, interp=1, fs=1, phat=True, t_max=None):
#     """
#     This function computes the time difference of arrival (TDOA)
#     of the signal at the two microphones. This in turns is used to infer
#     the direction of arrival (DOA) of the signal.
#     Specifically if s(k) is the signal at the reference microphone and
#     s_2(k) at the second microphone, then for signal arriving with DOA
#     theta we have
#     s_2(k) = s(k - tau)
#     with
#     tau = fs*d*sin(theta)/c
#     where d is the distance between the two microphones and c the speed of sound.
#     We recover tau using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)
#     method. The reference is
#     Knapp, C., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay.
#     Parameters
#     ----------
#     x1 : nd-array
#         The signal of the reference microphone
#     x2 : nd-array
#         The signal of the second microphone
#     interp : int, optional (default 1)
#         The interpolation value for the cross-correlation, it can
#         improve the time resolution (and hence DOA resolution)
#     fs : int, optional (default 44100 Hz)
#         The sampling frequency of the input signal
#     Return
#     ------
#     theta : float
#         the angle of arrival (in radian (I think))
#     pwr : float
#         the magnitude of the maximum cross correlation coefficient
#     delay : float
#         the delay between the two microphones (in seconds)
#     """
#     # zero padded length for the FFT
#     n = x1.shape[-1] + x2.shape[-1] - 1
#     if n % 2 != 0:
#         n += 1

#     # Generalized Cross Correlation Phase Transform
#     # Used to find the delay between the two microphones
#     # up to line 71
#     X1 = fft.rfft(np.array(x1, dtype=np.float32), n=n, axis=-1)
#     X2 = fft.rfft(np.array(x2, dtype=np.float32), n=n, axis=-1)

#     if phat:
#         X1 /= np.abs(X1)
#         X2 /= np.abs(X2)

#     cc = fft.irfft(X1 * np.conj(X2), n=interp * n, axis=-1)

#     # maximum possible delay given distance between microphones

#     if t_max is None:
#         t_max = n // 2 + 1

#     # reorder the cross-correlation coefficients
#     cc = np.concatenate((cc[..., -t_max:], cc[..., :t_max]), axis=-1)

#     # import matplotlib.pyplot as plt

#     # t = np.arange(-t_max/fs, (t_max)/fs, 1/fs) * 1e6
#     # plt.plot(t, cc[15])
#     # plt.show()

#     # pick max cross correlation index as delay
#     tau = np.argmax(np.abs(cc), axis=-1)
#     tau -= t_max  # because zero time is at the center of the array

#     return tau / (fs * interp)


# from sklearn.utils.extmath import weighted_mode


# def framewise_gccphat(x, frame_dur, sr, window="tukey"):
#     TMAX = int(round(1e-3 * sr))
#     frame_width = int(round(frame_dur * sr))

#     # Total number of frames
#     T = 1 + (x.shape[-1] - 1) // frame_width

#     # Drop samples to get a multiple of frame size
#     if x.shape[-1] % T != 0:
#         x = x[..., -x.shape[-1] % T :]

#     assert x.shape[-1] % T == 0
#     frames = np.array(np.split(x, T, axis=-1))

#     window = signal.get_window(window, frame_width)
#     frames = frames * window

#     # Consider only frames that have energy above some threshold (ignore silence)
#     ENERGY_THRESHOLD = 5e-4
#     frame_energy = np.max(np.mean(frames**2, axis=-1) ** 0.5, axis=-1)
#     mask = frame_energy > ENERGY_THRESHOLD
#     frames = frames[mask]

#     fw_gccphat = tdoa2(frames[..., 0, :], frames[..., 1, :], fs=sr, t_max=TMAX)

#     # print(mask)
#     # print(fw_gccphat)
#     # print(frame_energy[mask])
#     itd = weighted_mode(fw_gccphat, frame_energy[mask], axis=-1)[0]
#     return itd[0]


# def fw_itd_diff(s_est, s_gt, sr, frame_duration=0.25):
#     """
#     Computes frame-wise delta ITD
#     """
#     # print("GT")
#     itd_gt = framewise_gccphat(s_gt, frame_duration, sr) * 1e6
#     # print("GT FW_ITD", itd_gt)
#     # print("EST")
#     itd_est = framewise_gccphat(s_est, frame_duration, sr) * 1e6
#     # print("EST FW_ITD", itd_est)
#     return np.abs(itd_est - itd_gt)


# def cal_interaural_error(predictions, targets, sr, debug=False):
#     """Compute ITD and ILD errors
#     input: (1, time, channel, speaker)
#     """

#     TMAX = int(round(1e-3 * sr))
#     EPS = 1e-8
#     s_target = targets[0]  # [T,E,C]
#     s_prediction = predictions[0]  # [T,E,C]

#     # ITD is computed with generalized cross-correlation phase transform (GCC-PHAT)
#     ITD_target = [
#         tdoa2(s_target[:, 0, i].cpu().numpy(), s_target[:, 1, i].cpu().numpy(), fs=sr, t_max=TMAX) * 10**6
#         for i in range(s_target.shape[-1])
#     ]
#     if debug:
#         print("TARGET ITD", ITD_target)

#     ITD_prediction = [
#         tdoa2(
#             s_prediction[:, 0, i].cpu().numpy(),
#             s_prediction[:, 1, i].cpu().numpy(),
#             fs=sr,
#             t_max=TMAX,
#         )
#         * 10**6
#         for i in range(s_prediction.shape[-1])
#     ]

#     if debug:
#         print("PREDICTED ITD", ITD_prediction)

#     ITD_error1 = np.mean(np.abs(np.array(ITD_target) - np.array(ITD_prediction)))
#     ITD_error2 = np.mean(np.abs(np.array(ITD_target) - np.array(ITD_prediction)[::-1]))
#     ITD_error = min(ITD_error1, ITD_error2)

#     # ILD  = 10 * log_10(||s_left||^2 / ||s_right||^2)
#     ILD_target_beforelog = torch.sum(s_target[:, 0] ** 2, dim=0) / (torch.sum(s_target[:, 1] ** 2, dim=0) + EPS)
#     ILD_target = 10 * torch.log10(ILD_target_beforelog + EPS)  # [C]
#     ILD_prediction_beforelog = torch.sum(s_prediction[:, 0] ** 2, dim=0) / (
#         torch.sum(s_prediction[:, 1] ** 2, dim=0) + EPS
#     )
#     ILD_prediction = 10 * torch.log10(ILD_prediction_beforelog + EPS)  # [C]

#     ILD_error1 = torch.mean(torch.abs(ILD_target - ILD_prediction))
#     ILD_error2 = torch.mean(torch.abs(ILD_target - ILD_prediction.flip(0)))
#     ILD_error = min(ILD_error1.item(), ILD_error2.item())

#     return ITD_error, ILD_error


# def compute_doa(mic_pos, s, sr, nfft=2048, num_sources=1):
#     # freq_range = [100, 20000]

#     X = pra.transform.stft.analysis(
#         s.T,
#         nfft,
#         nfft // 2,
#     )
#     X = X.transpose([2, 1, 0])

#     algo_names = ["SRP", "MUSIC", "FRIDA", "TOPS", "WAVES", "CSSM", "NormMUSIC"]

#     srp = pra.doa.algorithms["NormMUSIC"](mic_pos.T, sr, nfft, c=343, num_sources=num_sources)
#     srp.locate_sources(X)

#     values = srp.grid.values
#     phi = np.linspace(-np.pi, np.pi, 360)

#     values = np.roll(values, shift=180)

#     # plt.plot(phi * 180 / np.pi, values)
#     # plt.xlim([-90, 90])
#     # plt.show()

#     peak_idx = 90 + np.argmax(values[90:270])
#     return phi[peak_idx]


# def doa_diff(mic_pos, est, gt, sr):
#     doa_est = compute_doa(mic_pos, est, sr)
#     doa_gt = compute_doa(mic_pos, gt, sr)
#     return np.abs(doa_gt - doa_est)


# def gcc_phat(s_left, s_right, sr):
#     X = rfft(s_left)
#     Y = rfft(s_right)

#     Z = X * np.conj(Y)

#     y = irfft(np.exp(1j * np.angle(Z)))
#     center = (len(y) + 1) // 2
#     y = np.concatenate([y[center:], y[:center]])
#     lags = (np.linspace(0, len(y), len(y)) - ((len(y) + 1) / 2)) / sr
#     x = np.argmax(y)
#     tau = lags[x]

#     return lags, y


# def gcc_phat_diff(s_est, s_gt, sr):
#     TMAX = int(round(1e-3 * sr))
#     itd_est = tdoa2(s_est[..., 0, :], s_est[..., 1, :], fs=sr, t_max=TMAX)
#     itd_gt = tdoa2(s_gt[..., 0, :], s_gt[..., 1, :], fs=sr, t_max=TMAX)
#     return np.abs(itd_est - itd_gt) * 10**6


# def si_sdr(estimated_signal, reference_signals, scaling=True):
#     """
#     This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
#     or https://github.com/sigsep/bsseval/issues/3 for the motivation and
#     explanation
#     Input:
#         estimated_signal and reference signals are (N,) numpy arrays
#     Returns: SI-SDR as scalar
#     """

#     Rss = np.dot(reference_signals, reference_signals)
#     this_s = reference_signals

#     if scaling:
#         # get the scaling factor for clean sources
#         a = np.dot(this_s, estimated_signal) / Rss
#     else:
#         a = 1

#     e_true = a * this_s
#     e_res = estimated_signal - e_true

#     Sss = (e_true**2).sum()
#     Snn = (e_res**2).sum()

#     SDR = 10 * np.log10(Sss / Snn)

#     return SDR
