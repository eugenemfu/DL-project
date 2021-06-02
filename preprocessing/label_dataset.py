import argparse
import scipy.io
import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('dataset_path',
                        type=lambda value: Path(value).absolute(),
                        help='Absolute path to the Annotations.mat file. Output file with image_names and emotions'
                             'will be created in the same directory where is Annotations.mat file.')


def main(data_path: str):
    for holdout in ['train', 'val', 'test']:
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

        df.fillna('No emotion', inplace=True)

        emotion_1_unique = df['emotion_1'].unique()
        emotion_2_unique = df['emotion_2'].unique()

        emotion_unique = list(emotion_1_unique)
        for emotion in emotion_2_unique:
            emotion_unique.append(emotion)
        emotion_unique = np.unique(emotion_unique)

        le = LabelEncoder()
        le.fit(emotion_unique)

        encoded_emotion_1 = le.transform(df['emotion_1'])
        encoded_emotion_2 = le.transform(df['emotion_2'])

        encoded_emotion = []
        for em1, em2 in zip(encoded_emotion_1, encoded_emotion_2):
            encoded_emotion.append([em1, em2])

        mlb = MultiLabelBinarizer()
        encoded_targets = mlb.fit_transform(encoded_emotion)
        ids = df['id']
        targets = pd.DataFrame(encoded_targets)
        targets.insert(0, "id", ids.values, True)

        output_path = Path(data_path).parent / f'{holdout}.csv'
        targets.to_csv(output_path, index=False)

    # save emotion-keys
    df = pd.DataFrame(
        {"id": range(emotion_unique.shape[0]),
         "emotion": le.classes_
         }
    )
    df.to_csv(Path(data_path).parent / 'emotion_keys.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    main(args.dataset_path)
