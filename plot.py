#!/bin/python3
import plotly.graph_objects as go
import pandas as pd


def plot_labels_dist(filename: str) -> None:
    df = pd.read_csv(filename)
    labels = df['label']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=labels, marker_color='#330C73', opacity=0.9))
    fig.update_layout(title_text='Image Emotion Count', xaxis_title_text='Emotion',  yaxis_title_text='Count', bargap=0.2)
    fig.show()


plot_labels_dist('data_resized/train.csv')
