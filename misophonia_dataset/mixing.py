from collections.abc import Collection

import eliot
import numpy as np

from ._binamix import custom_mix_tracks_binaural, setup_binamix
from .interface import GlobalMixingParams, SourceDataItem, SourceTrack

setup_binamix()
from binamix.sadie_utilities import TrackObject  # type: ignore  # noqa: I001


TrackAudioSpec = tuple[SourceTrack, np.ndarray]


def prepare_track_specs(
    fg_items: Collection[SourceDataItem],
    bg_items: Collection[SourceDataItem],
    global_params: GlobalMixingParams,
    *,
    fg_track_options: dict | None = None,
    bg_track_options: dict | None = None,
    rng: np.random.Generator | None = None,
    max_length: int = 308700,  # 7 seconds at 44.1 kHz
) -> tuple[tuple[TrackAudioSpec, ...], tuple[TrackAudioSpec, ...]]:
    """
    Prepares track specifications for binaural mixing for foreground and background audios. Locates meaningful audio in background sounds to use in mix.
    If foreground sound is longer than max_length (7 sec), then it will only include the most meaningful 7 seconds of audio.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _generate_track_specs(
        item: SourceDataItem, audio: np.ndarray, fg_max_len: int, options: dict
    ) -> TrackAudioSpec:

        def _locate_meaningful_audio(
            audio: np.ndarray, fg_max_len: int, last_place_to_start_audio: int
        ) -> tuple[int, int]:
            """
            Takes an an array of audio, the length of the clip to be used in the mix, and the last place to start.
            Finds the start and end of a clip of length fg_max_len that contains the most energy.

            Returns:
                indices of start and end of clip to be used in the mix
            """
            step_size = 100
            n = len(audio)
            x = audio.astype(np.float64, copy=False)

            sq = x * x
            csum = np.empty(n + 1, dtype=np.float64)
            csum[0] = 0.0
            csum[1:] = np.cumsum(sq)

            candidate_starts = np.arange(0, last_place_to_start_audio + 1, step_size)
            candidate_ends = candidate_starts + fg_max_len

            energies = csum[candidate_ends] - csum[candidate_starts]

            best_idx = np.argmax(energies)
            # best_energy = energies[best_idx]

            best_start = int(candidate_starts[best_idx])
            return best_start, best_start + fg_max_len

        length = audio.shape[0]
        if length == fg_max_len:
            start = 0  # No need to start at a random place
            end = fg_max_len
        elif length > fg_max_len:
            # Crop longer clips to be the same length as fg_max_len
            last_place_to_start_audio = length - fg_max_len
            audio_start, audio_end = _locate_meaningful_audio(audio, fg_max_len, last_place_to_start_audio)
            audio = audio[audio_start:audio_end]

            start = 0
            end = fg_max_len
        else:
            # Find placement at random (rest will be zero padded)
            last_place_to_pad = fg_max_len - length
            start = rng.integers(0, last_place_to_pad + 1)
            end = start + length

        assert end - start <= fg_max_len, "Audio exceeds max length after placement. This should never happen."

        track = SourceTrack(
            source_item=item,
            start=start,
            end=end,
            _rng=rng,
            **options,
        )

        return track, audio

    fg_audios = tuple((item, item.load_audio(sample_rate=global_params.sample_rate)[0]) for item in fg_items)
    bg_audios = tuple((item, item.load_audio(sample_rate=global_params.sample_rate)[0]) for item in bg_items)

    # Either longest foreground audio or max_length, whichever is shorter
    fg_max_length = max(audio.shape[0] for _, audio in fg_audios)
    fg_max_length = min(fg_max_length, max_length)

    fg_specs = tuple(
        _generate_track_specs(item, audio, fg_max_length, fg_track_options or {}) for item, audio in fg_audios
    )
    bg_specs = tuple(
        _generate_track_specs(item, audio, fg_max_length, bg_track_options or {}) for item, audio in bg_audios
    )

    return fg_specs, bg_specs


def binaural_mix(
    fg_specs: tuple[TrackAudioSpec, ...],
    bg_specs: tuple[TrackAudioSpec, ...],
    global_params: GlobalMixingParams,
    target_snr_db: float,
    target_snr_range: tuple[float, float] = (5.0, 10.0),
    *,
    is_trig: bool,
    max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Max a binaural mix of a foreground (trigger) and background sound.

    Returns a tuple of (mix, ground_truth, clean_mix), where:
    - mix is the binaural mix of the foreground and background sounds.
    - isolated_trigger is the binaural audio of the isolated trigger sound
    - clean_mix is the binaural mix of the background sounds without any triggers
    """

    fg_specs = tuple(fg_specs)
    bg_specs = tuple(bg_specs)

    fg_specs, bg_specs = _normalize_and_pad(fg_specs, bg_specs)

    def _make_binamix_track(spec: TrackAudioSpec) -> TrackObject:
        track, padded_audio = spec
        return TrackObject(
            name=track.source_item.file_path.stem,
            azimuth=track.azimuth,
            elevation=track.elevation,
            level=track.level,
            reverb=track.reverb,
            audio=padded_audio,
        )

    fg_binamix_tracks = list(map(_make_binamix_track, fg_specs))
    bg_binamix_tracks = list(map(_make_binamix_track, bg_specs))

    if not is_trig:
        # For control sounds, we want isolated_trigger and clean_mix to be None, since they are the same as the mix
        mix = custom_mix_tracks_binaural(
            tracks=[*fg_binamix_tracks, *bg_binamix_tracks],
            subject_id=global_params.subject_id,
            sample_rate=global_params.sample_rate,
            ir_type=global_params.ir_type,
            speaker_layout=global_params.speaker_layout,
            mode=global_params.mode,
            reverb_type=global_params.reverb_type,
        )
        mix = _ensure_max_length(mix, max_length)

        return mix, None, None

    # For triggers, we need to mix the isolated trigger and clean mix separately:
    isolated_trigger = custom_mix_tracks_binaural(
        tracks=fg_binamix_tracks,
        subject_id=global_params.subject_id,
        sample_rate=global_params.sample_rate,
        ir_type=global_params.ir_type,
        speaker_layout=global_params.speaker_layout,
        mode=global_params.mode,
        reverb_type=global_params.reverb_type,
    )
    isolated_trigger = _ensure_max_length(isolated_trigger, max_length)

    clean_background = custom_mix_tracks_binaural(
        tracks=bg_binamix_tracks,
        subject_id=global_params.subject_id,
        sample_rate=global_params.sample_rate,
        ir_type=global_params.ir_type,
        speaker_layout=global_params.speaker_layout,
        mode=global_params.mode,
        reverb_type=global_params.reverb_type,
    )
    clean_background = _ensure_max_length(clean_background, max_length)

    assert isolated_trigger.shape == isolated_trigger.shape

    trigger_power = np.mean(isolated_trigger**2)
    clean_background_power = np.mean(clean_background**2)
    alpha = snr_control_scaling_factor(trigger_power, clean_background_power, target_snr_db=target_snr_db)

    scaled_clean_background = clean_background * alpha

    calculated_snr_db = _calculate_snr(isolated_trigger, scaled_clean_background)
    if not (target_snr_range[0] - 1 <= calculated_snr_db <= target_snr_range[1] + 1):
        eliot.log_message(
            f"SNR is outside the target range after scaling. Calculated SNR: {calculated_snr_db:.2f} dB, Target range: {target_snr_range} dB",
            level="warning",
            to_stderr=True,  # Stdout is supressed by Binamix quickfix
        )

    # Added small tolerance since their may be numerical imprecision for tiny background power.
    mix = isolated_trigger + scaled_clean_background
    mix = _ensure_max_length(mix, max_length)

    return mix, isolated_trigger, scaled_clean_background


def _ensure_max_length(audio: np.ndarray, max_length: int | None) -> np.ndarray:
    # Assumes (C, T) audio
    assert audio.ndim == 2
    assert audio.shape[0] == 2
    if max_length is None or audio.shape[1] <= max_length:
        return audio
    return audio[:, :max_length]


def _normalize_and_pad(
    fg_tracks: tuple[TrackAudioSpec, ...],
    bg_tracks: tuple[TrackAudioSpec, ...],
) -> tuple[tuple[TrackAudioSpec, ...], tuple[TrackAudioSpec, ...]]:

    fg_max_end = max(track.end for track, _ in fg_tracks)
    # Pad in case that the audio is shorter than max fg audio
    fg_padded = tuple((track, np.pad(audio, (track.start, fg_max_end - track.end))) for track, audio in fg_tracks)
    bg_padded = tuple((track, np.pad(audio, (track.start, fg_max_end - track.end))) for track, audio in bg_tracks)

    assert all(len(audio) == fg_max_end for _, audio in fg_padded + bg_padded)

    return fg_padded, bg_padded


def snr_control_scaling_factor(fg_power: float, bg_power: float, target_snr_db: float) -> float:
    """
    Find scaling factor to achieve desired relative signal between foreground and backgrounds.
    """
    eps = np.finfo(np.float32).tiny
    fg_power = fg_power + eps
    bg_power = bg_power + eps

    # Solve for a single background scaling factor alpha such that:
    # 10 * log10(fg_power / (alpha^2 * bg_power)) = target_snr_db
    desired_bg_power = fg_power / (10 ** (target_snr_db / 10.0))
    alpha = np.sqrt(desired_bg_power / bg_power)

    return alpha


def _calculate_snr(fg: np.ndarray, bg: np.ndarray) -> float:
    """
    Calculate the SNR in dB between a foreground and background signal. Helper function to ensure that SNR target range
    is being applied as intended.
    """
    eps = np.finfo(np.float32).tiny
    fg_power = np.mean(fg**2) + eps
    bg_power = np.mean(bg**2) + eps
    snr_db = 10 * np.log10(fg_power / bg_power)
    return snr_db
