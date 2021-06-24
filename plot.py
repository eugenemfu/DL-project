#!/bin/python3
import plotly.graph_objects as go
import pandas as pd


def plot_labels_dist(filename: str = 'data64/train.csv') -> None:
    df = pd.read_csv(filename)
    labels = df['label']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=labels, marker_color='#330C73', opacity=0.9))
    fig.update_layout(title_text='Image Emotion Count', xaxis_title_text='Emotion',  yaxis_title_text='Count', bargap=0.2)
    fig.show()


def plot_images(filename: str = 'data64/train.csv') -> None:
    df = pd.read_csv(filename)
    dictionary_to_show = {'0': False, '1': False, '2': False, '3': False, '4': False, '5': False, '6': False}
    for j in range(7):
        for i in range(df.shape[0]):
            if int(df.iloc[i, 1]) == int(j) and dictionary_to_show[str(j)] is False:
                dictionary_to_show[str(j)] = df.iloc[i, 0]
                break
    images = list(dictionary_to_show.values())
    # print(Image.open(images[6]).show())


import numpy as np

def plot_labels_dist(filename: str = 'data64/train.csv') -> None:
    x = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    df = pd.read_csv(filename)
    labels = df['label'].to_numpy()
    labels = [len(np.where(labels == i)[0]) for i in range(7)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=labels, marker_color='#330C73', opacity=0.9, name='real data'))
    new_labels = [0, 4000 + labels[1], 0, 0, 0, 0, 0]
    fig.add_trace(go.Bar(x=x, y=new_labels, marker_color='#EF553B', opacity=0.9, name='augmented data'))

    fig.update_layout(barmode='group', title_text='Image Emotion Count')
    fig.show()

# plot_labels_dist()


def plot_f1_(filename: str = 'data64/test.csv') -> None:
    x = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    df = pd.read_csv(filename)
    labels = df['label'].to_numpy()
    labels = [len(np.where(labels == i)[0]) / len(labels) for i in range(7)]
    fig = go.Figure()

    # fr_aug = [0.46, 1, 0.76, 0.17, 0.47, 0.41, 0.23]
    # fa_aug = [0.63, 0.0, 0.38, 0.15, 0.36, 0.8, 0.26]
    # f1_aug = [0.49, 0.0, 0.3, 0.84, 0.56, 0.49, 0.75]

    fr_aug_g = [0.552, 0.676, 0.617, 0.206, 0.420, 0.501, 0.266]
    fa_aug_g = [0.397, 0.099, 0.483, 0.193, 0.594, 0.656, 0.225]
    f1_aug_g = [0.486, 0.456, 0.410, 0.799, 0.534, 0.463, 0.749]

    fig.add_trace(go.Bar(x=x, y=labels, opacity=0.9, name='share'))
    fig.add_trace(go.Bar(x=x, y=fa_aug_g, opacity=0.9, name='false alarm'))
    fig.add_trace(go.Bar(x=x, y=fr_aug_g, opacity=0.9, name='false reject'))
    fig.add_trace(go.Bar(x=x, y=f1_aug_g, opacity=0.9, name='f1 score'))


    fig.update_layout(barmode='group', title_text='F1 score simple augmentation')
    fig.show()

# plot_f1_()
