#!/bin/python3

import os
import pandas as pd


labels = {
	'angry': 0,
	'disgust': 1,
	'fear': 2,
	'happy': 3,
	'neutral': 4,
	'sad': 5,
	'surprise': 6,
}


def make_csv(data_path, csv_filename):
	path_list = []
	label_list = []
	for dirpath, _, filenames in os.walk(data_path):
		if dirpath != data_path:
			label = labels[dirpath[len(data_path)+1:]]
			for filename in filenames:
				path = os.path.join(dirpath, filename)
				path_list.append(path)
				label_list.append(label)
	df = pd.DataFrame({'path': path_list, 'label': label_list})
	df.to_csv(csv_filename, index=False)


folder = 'data64aug'
make_csv(folder + '/train', folder + '/train.csv')
make_csv(folder + '/test', folder + '/test.csv')
