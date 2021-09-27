from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ._const import NP_RANDOM_SEED

np.random.seed(NP_RANDOM_SEED)


class MCV:
    def __init__(
        self,
        base_path: Path,
        val_dataset_size: int,
    ) -> None:
        self._base_path: Path = base_path
        self._val_dataset_size: int = val_dataset_size

    def get_train_val_filenames(
        self,
    ) -> Tuple[List[str], List[str]]:
        files = [str(self._base_path / 'clips' / filename) for filename in self._get_filenames('train.tsv')]

        train = files[:-self._val_dataset_size]
        val = files[-self._val_dataset_size:]

        print(f'Train samples: {len(train)} | Val samples: {len(val)}')

        return train, val

    def get_test_filenames(
        self,
    ) -> List[str]:
        test = [str(self._base_path / 'clips' / filename) for filename in self._get_filenames('test.tsv')]

        print(f'Test samples: {len(test)}')

        return test

    def _get_filenames(
        self,
        df_name: str,
    ) -> np.ndarray:
        print('Getting MCV metadata...')

        metadata = pd.read_csv(
            self._base_path / df_name,
            sep='\t',
        )

        files = metadata['path'].values
        np.random.shuffle(files)

        return files
