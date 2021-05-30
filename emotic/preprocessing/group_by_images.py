import argparse
import shutil
import os
import pandas as pd
from pathlib import Path


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('folder_path',
                        type=lambda value: Path(value).absolute(),
                        help='Absolute path to the folder contained all Emotic images.')

    parser.add_argument('target_path',
                        type=lambda value: Path(value).absolute(),
                        help='Path to the csv file get by label_dataset.py script.')

    parser.add_argument('holdout',
                        type=str,
                        help='train, val or test – should correspond to the train, val or test labels in csv file.')


def main(folder_path: str, target_path: str, holdout: str):
    image_names = pd.read_csv(target_path)['id']
    output_dir_path = os.path.join(Path(folder_path).parent, holdout)
    os.mkdir(output_dir_path)
    for image in image_names:
        shutil.copy(Path(folder_path) / image, output_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    main(args.folder_path, args.target_path, args.holdout)
