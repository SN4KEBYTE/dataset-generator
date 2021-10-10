import multiprocessing as mp
import warnings
from pathlib import Path
from typing import (
    List,
    Optional,
)

import numpy as np
import soundfile as sf

from ._audio import add_noise, read_audio
from ._const import NP_RANDOM_SEED

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
)

np.random.seed(NP_RANDOM_SEED)


class Dataset:
    def __init__(
        self,
        out_path: Path,
        sample_rate: int,
        cores: Optional[int] = None,
    ) -> None:
        self._out_path: Path = out_path
        self._sample_rate = sample_rate
        self._cores = cores if cores else mp.cpu_count()

    def create(
        self,
        clean_filenames: List[str],
        noise_filenames: List[str],
    ) -> None:
        samples = (
            (
                clean_file,
                read_audio(
                    np.random.choice(noise_filenames),
                    self._sample_rate,
                )[0],
            )
            for clean_file in clean_filenames
        )

        print('Starting processing...')
        q = mp.Queue(maxsize=self._cores)
        pool = mp.Pool(
            self._cores,
            initializer=self._process_file,
            initargs=(q,),
        )

        for samp in samples:
            q.put(samp)

        for _ in range(self._cores):
            q.put(None)

        pool.close()
        pool.join()

    def _process_file(
        self,
        q: mp.Queue,
    ) -> None:
        while True:
            try:
                clean_file, noise = q.get()
            except TypeError:
                break

            clean_audio, _ = read_audio(
                clean_file,
                self._sample_rate,
            )
            noised = add_noise(
                clean_audio,
                noise,
            )

            sf.write(
                self._out_path / (Path(clean_file).stem + '.wav'),
                noised,
                self._sample_rate,
                format='WAV',
            )
