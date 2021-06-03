import argparse
import scipy.io
import pandas as pd
from pathlib import Path
import os

from sklearn.preprocessing import LabelEncoder


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('dataset_path',
                        type=lambda value: Path(value).absolute(),
                        help='Absolute path to the Annotations.mat file. Output file with image_names and emotions'
                             'will be created in the same directory where is Annotations.mat file.')


def main(data_path: str):
    
    try:
        os.mkdir('labels')
    # pass if directory a;ready exists
    except OSError:
        pass
    
    for holdout in ['train', 'val', 'test']:
        mat = scipy.io.loadmat(data_path)
        image_names, first_emotion = [], []
        for image in range(len(mat[holdout][0])):
            image_names.append(mat[holdout][0][image][0][0])
            first_emotion.append(mat[holdout][0][image][4][0][0][1][0][0][0][0][0][0])

        df = pd.DataFrame(
            {"id": image_names,
             "emotion": first_emotion
             }
        )

        emotion_unique = list(df['emotion'].unique())

        le = LabelEncoder()
        le.fit(emotion_unique)

        encoded_emotion = le.transform(df['emotion'])

        df.insert(2, "emotion_id", encoded_emotion, True)

        output_path = Path('labels') / f'{holdout}.csv'
        df.to_csv(output_path, index=False)

    # save emotion-keys
    df = pd.DataFrame(
        {"id": range(len(emotion_unique)),
         "emotion": le.classes_
         }
    )
    df.to_csv(Path('labels') / 'emotion_keys.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    main(args.dataset_path)
