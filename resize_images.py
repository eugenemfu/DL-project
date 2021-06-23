#!/bin/python3

import os
import cv2
import typing


SIZE = 64


def reshape_and_save(source_path: str, output_path: str):
	img = cv2.imread(source_path)[:, :, 0]
	scale = SIZE / min(img.shape[0], img.shape[1])
	width = int(round(img.shape[1] * scale))
	height = int(round(img.shape[0] * scale))
	resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
	cropped = resized[(height-SIZE)//2:(height+SIZE)//2, (width-SIZE)//2:(width+SIZE)//2]
	cv2.imwrite(output_path, cropped)


def reshape_dataset(data_path, output_path):
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	i = 0
	print(f'\rReshaped {i} images', end='')
	for dirpath, dirnames, filenames in os.walk(data_path):
		for dirname in dirnames:
			new_dirpath = os.path.join(output_path, dirpath[len(data_path)+1:], dirname)
			#print(new_dirpath)
			if not os.path.exists(new_dirpath):
				os.mkdir(new_dirpath)
		for filename in filenames:
			if filename.endswith('.jpg'):
				source_path = os.path.join(dirpath, filename)
				output_file = os.path.join(output_path, dirpath[len(data_path)+1:], filename)
				reshape_and_save(source_path, output_file)
				i += 1
				print(f'\rReshaped {i} images', end='')
	

reshape_dataset('data', 'data64')
