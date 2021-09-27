from pathlib import Path
from typing import (
    List,
    Tuple,
)

import numpy as np
import pandas as pd

from ._const import NP_RANDOM_SEED

np.random.seed(NP_RANDOM_SEED)


class UrbanSound8k:
    def __init__(
        self,
        base_path: Path,
        val_dataset_size: int,
        class_ids: np.ndarray = None,
    ) -> None:
        self._base_path = base_path
        self._val_dataset_size = val_dataset_size
        self._class_ids = class_ids
        self._metadata = self._get_metadata()

    def _get_metadata(
        self,
    ) -> pd.DataFrame:
        metadata = pd.read_csv(self._base_path / 'UrbanSound8K.csv')
        metadata.reindex(np.random.permutation(metadata.index))

        return metadata

    def _get_filenames_by_class_id(
        self,
        metadata: pd.DataFrame,
    ) -> List[str]:
        if self._class_ids is None:
            self._class_ids = np.unique(self._metadata['classID'].values)

        files = []

        for class_id in self._class_ids:
            per_id_files = metadata[metadata['classID'] == class_id][['slice_file_name', 'fold']].values
            files.extend([str(self._base_path / ('fold' + str(file[1])) / file[0]) for file in per_id_files])

        return files

    def get_train_val_filenames(
        self,
    ) -> Tuple[List[str], List[str]]:
        train_meta = self._metadata[self._metadata.fold != 10]

        filenames = self._get_filenames_by_class_id(train_meta)
        np.random.shuffle(filenames)

        val = filenames[-self._val_dataset_size:]
        train = filenames[:-self._val_dataset_size]

        print(f'Train samples: {len(train)} | Val samples: {len(val)}')

        return train, val

    def get_test_filenames(
        self,
    ) -> List[str]:
        test_meta = self._metadata[self._metadata.fold == 10]

        test = self._get_filenames_by_class_id(test_meta)
        np.random.shuffle(test)

        print(f'Test samples: {len(test)}')

        return test
