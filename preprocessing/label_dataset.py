import argparse
import scipy.io
import pandas as pd
from pathlib import Path


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('dataset_path',
                        type=lambda value: Path(value).absolute(),
                        help='Absolute path to the Annotations.mat file. Output file with image_names and emotions'
                             'will be created in the same directory where is Annotations.mat file.')

    parser.add_argument('holdout',
                        type=str,
                        help='Annotations.mat file is divided into: train, val and test part, so, holdout'
                             'has to be one of these values.')


def main(data_path: str, holdout: str):
    mat = scipy.io.loadmat(data_path)
    image_names, first_emotion, second_emotion = [], [], []
    for image in range(len(mat[holdout][0])):
        image_names.append(mat[holdout][0][image][0][0])
        first_emotion.append(mat[holdout][0][image][4][0][0][1][0][0][0][0][0][0])
        try:
            second_emotion.append(mat[holdout][0][image][4][0][0][1][0][0][0][0][1][0])
        except IndexError:
            # not all samples have 2 emotions in dataset
            second_emotion.append(None)
    df = pd.DataFrame(
        {"id": image_names,
         "emotion_1": first_emotion,
         "emotion_2": second_emotion,
         }
    )
    output_path = Path(data_path).parent / f'emotic_{holdout}_labels.csv'
    df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    main(args.dataset_path, args.holdout)
