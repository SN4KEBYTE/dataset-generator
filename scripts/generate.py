from argparse import ArgumentParser
from pathlib import Path
from sys import (
    exit,
    stderr,
)
from typing import List

from src import (
    Dataset,
    MCV,
    UrbanSound8k,
)

_DESCRIPTION = """
Generate noised audio for noise reduction models training.
Based on UrbanSound8k and Mozilla Common Voice datasets
"""


def _validate_dir(
    path: Path,
) -> bool:
    return path.exists() and path.is_dir()


def _create_set(
    out_path: Path,
    mcv_filenames: List[str],
    u8k_filenames: List[str],
    **kwargs,
) -> None:
    ds = Dataset(
        out_path, **kwargs,
    )
    ds.create(
        mcv_filenames,
        u8k_filenames,
    )


if __name__ == '__main__':
    parser = ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        '--mcv',
        type=str,
        help='Directory with MCV dataset',
    )
    parser.add_argument(
        '--urban8k',
        type=str,
        help='--Directory with UrbanSound8K dataset',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Directory for resulting files (must be empty)',
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sample rate',
    )
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of cores to be used when generating dataset',
    )
    parser.add_argument(
        '--mcv_val_size',
        type=int,
        default=1000,
        help='Number of samples in validation set for Mozilla Common Voice dataset'
    )
    parser.add_argument(
        '--u8k_val_size',
        type=int,
        default=200,
        help='Number of samples in validation set for UrbanSound8k dataset'
    )

    args = parser.parse_args()
    mcv_dir = Path(args.mcv)
    urban8k_dir = Path(args.urban8k)
    out_dir = Path(args.out)

    sample_rate = args.sr
    cores = args.cores

    if not _validate_dir(mcv_dir):
        stderr.write('Path to MCV doesn\'t exist or isn\'t a directory')
        exit(1)

    if not _validate_dir(urban8k_dir):
        stderr.write('Path to UrbanSound8K doesn\'t exist or isn\'t a directory')
        exit(1)

    if not _validate_dir(out_dir) and not len(list(out_dir.iterdir())):
        stderr.write('Out directory doesn\'t exist or isn\'t a directory or isn\'t empty')
        exit(1)

    print('Getting MCV train, val and test filenames...')
    mcv = MCV(
        mcv_dir,
        args.mcv_val_size,
    )
    mcv_train_filenames, mcv_val_filenames = mcv.get_train_val_filenames()
    mcv_test_filenames = mcv.get_test_filenames()

    print('Getting U8K train, val and test filenames...')
    u8k = UrbanSound8k(
        urban8k_dir,
        args.u8k_val_size,
    )
    u8k_train_filenames, u8k_val_filenames = u8k.get_train_val_filenames()
    u8k_test_filenames = u8k.get_test_filenames()

    train_out_dir = out_dir / 'train'
    train_out_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    print('Applying noise to train data...')
    _create_set(
        train_out_dir,
        mcv_train_filenames,
        u8k_train_filenames,
        sample_rate=sample_rate,
        cores=cores,
    )

    val_out_dir = out_dir / 'val'
    val_out_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    print('Applying noise to val data...')
    _create_set(
        val_out_dir,
        mcv_val_filenames,
        u8k_val_filenames,
        sample_rate=sample_rate,
        cores=cores,
    )

    test_out_dir = out_dir / 'test'
    test_out_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    print('Applying noise to test data...')
    _create_set(
        test_out_dir,
        mcv_test_filenames,
        u8k_test_filenames,
        sample_rate=sample_rate,
        cores=cores,
    )

    print('DONE')
