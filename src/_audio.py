from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import librosa
import numpy as np


def read_audio(
    path: str,
    sample_rate: Optional[int],
    normalize: bool = True,
    **kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(
        path,
        sample_rate,
        **kwargs,
    )

    if normalize:
        librosa.util.normalize(audio)

    return audio, sr


def add_noise(
    clean_audio: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    while clean_audio.size >= noise.size:
        noise = np.append(noise, noise)

    i = np.random.randint(
        0,
        noise.size - clean_audio.size,
    )
    noise_segment = noise[i:i + clean_audio.size]

    clean_power = np.sum(np.square(clean_audio))
    noise_power = np.sum(np.square(noise_segment))

    return clean_audio + np.sqrt(clean_power / noise_power) * noise_segment
