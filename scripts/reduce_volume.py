from argparse import ArgumentParser
from pathlib import Path
from sys import (
    exit,
    stderr,
)

import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

_DESCRIPTION = """
Reduce volume for all WAV files contained in directory
"""


def _reduce_volume(
    in_dir: Path,
    reduce_level: int,
) -> None:
    for file in tqdm(in_dir.iterdir()):
        if file.is_dir():
            _reduce_volume(
                file,
                reduce_level,
            )
        elif file.suffix == '.wav':
            # fixing encoding
            audio, sr = sf.read(file)
            sf.write(
                file,
                audio,
                sr,
            )

            audio = AudioSegment.from_file(file)
            audio -= reduce_level

            audio.export(
                file,
                'wav',
            )


if __name__ == '__main__':
    parser = ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        '--in_dir',
        type=str,
        help='Input directory',
    )
    parser.add_argument(
        '--reduce_level',
        type=int,
        default=15,
        help='Reduction level in dB',
    )

    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    reduce_level = args.reduce_level

    if not in_dir.is_dir() or not in_dir.exists():
        stderr.write('Input directory does not exist or is not a directory')
        exit(1)

    if reduce_level <= 0:
        stderr.write('Reduction level must be positive number')
        exit(1)

    _reduce_volume(
        in_dir,
        reduce_level,
    )
