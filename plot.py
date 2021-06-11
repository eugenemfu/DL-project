#!/bin/python3
import plotly.graph_objects as go
import pandas as pd
from PIL import Image


def plot_labels_dist(filename: str = 'data_resized/train.csv') -> None:
    df = pd.read_csv(filename)
    labels = df['label']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=labels, marker_color='#330C73', opacity=0.9))
    fig.update_layout(title_text='Image Emotion Count', xaxis_title_text='Emotion',  yaxis_title_text='Count', bargap=0.2)
    fig.show()


def plot_images(filename: str = 'data_resized/train.csv') -> None:
    df = pd.read_csv(filename)
    dictionary_to_show = {'0': False, '1': False, '2': False, '3': False, '4': False, '5': False, '6': False}
    for j in range(7):
        for i in range(df.shape[0]):
            if int(df.iloc[i, 1]) == int(j) and dictionary_to_show[str(j)] is False:
                dictionary_to_show[str(j)] = df.iloc[i, 0]
                break
    images = list(dictionary_to_show.values())
    # print(Image.open(images[6]).show())
