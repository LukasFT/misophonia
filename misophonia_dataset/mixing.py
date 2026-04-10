from collections.abc import Collection

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
) -> tuple[tuple[TrackAudioSpec, ...], tuple[TrackAudioSpec, ...]]:
    if rng is None:
        rng = np.random.default_rng()

    def _generate_track_specs(
        item: SourceDataItem, audio: np.ndarray, fg_max_len: int, options: dict
    ) -> TrackAudioSpec:

        def _locate_meaningful_audio(audio: np.ndarray, fg_max_len: int, last_place_to_start_audio: int) ->tuple[int, int]:
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
            best_energy = energies[best_idx]

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

    fg_max_length = max(audio.shape[0] for _, audio in fg_audios)
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
    *,
    is_trig: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Max a binaural mix of a foreground (trigger) and background sound.
    """

    fg_specs = tuple(fg_specs)
    bg_specs = tuple(bg_specs)

    fg_specs, bg_specs = _normalize_and_pad(fg_specs, bg_specs)

    def _make_binamix_track(spec: TrackAudioSpec) -> TrackObject:
        track, padded_audio = spec
        return TrackObject(  # NOTE: Tonio change this, but it should not be changed
            name=track.source_item.file_path.stem,
            azimuth=track.azimuth,
            elevation=track.elevation,
            level=track.level,
            reverb=track.reverb,
            audio=padded_audio,
        )

    fg_binamix_tracks = list(map(_make_binamix_track, fg_specs))
    bg_binamix_tracks = list(map(_make_binamix_track, bg_specs))

    mix = custom_mix_tracks_binaural(
        tracks=[*fg_binamix_tracks, *bg_binamix_tracks],
        subject_id=global_params.subject_id,
        sample_rate=global_params.sample_rate,
        ir_type=global_params.ir_type,
        speaker_layout=global_params.speaker_layout,
        mode=global_params.mode,
        reverb_type=global_params.reverb_type,
    )

    if is_trig:
        ground_truth = custom_mix_tracks_binaural(
            tracks=fg_binamix_tracks,
            subject_id=global_params.subject_id,
            sample_rate=global_params.sample_rate,
            ir_type=global_params.ir_type,
            speaker_layout=global_params.speaker_layout,
            mode=global_params.mode,
            reverb_type=global_params.reverb_type,
        )
        assert ground_truth.shape == mix.shape, "Ground truth and mix shapes do not match."

        return mix, ground_truth
    else:
        return mix, None  # silence for control sound


def _normalize_and_pad(
    fg_tracks: tuple[TrackAudioSpec, ...],
    bg_tracks: tuple[TrackAudioSpec, ...],
) -> tuple[tuple[TrackAudioSpec, ...], tuple[TrackAudioSpec, ...]]:
    # RMS normalization:
    # rms_fg = [np.sqrt(np.mean(audio**2)) for _, audio in fg_tracks]
    # rms_bg = [np.sqrt(np.mean(audio**2)) for _, audio in bg_tracks]
    # rms_target = np.mean(rms_fg + rms_bg)
    # fg_norm = tuple(
    #     (item, audio * (rms_target / rms)) if rms > 1e-6 else (item, audio)
    #     for (item, audio), rms in zip(fg_tracks, rms_fg)
    # )
    # bg_norm = tuple(
    #     (item, audio * (rms_target / rms)) if rms > 1e-6 else (item, audio)
    #     for (item, audio), rms in zip(bg_tracks, rms_bg)
    # )

    fg_max_end = max(track.end for track, _ in fg_tracks)
    # Pad in case that the audio is shorter than max fg audio
    fg_padded = tuple((track, np.pad(audio, (track.start, fg_max_end - track.end))) for track, audio in fg_tracks)
    bg_padded = tuple((track, np.pad(audio, (track.start, fg_max_end - track.end))) for track, audio in bg_tracks)


    assert all(len(audio) == fg_max_end for _, audio in fg_padded + bg_padded)
    return fg_padded, bg_padded
