#!/bin/python3

import os
import cv2
import typing


SIZE = 128


def reshape_and_save(source_path: str, output_path: str):
	img = cv2.imread(source_path)
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
	print(f'Reshaped {i} images out of 23554', end='\r')
	for dirpath, _, filenames in os.walk(data_path):
		for filename in filenames:
			if filename.endswith('.jpg'):
				source_path = os.path.join(dirpath, filename)
				output_file = os.path.join(output_path, filename)
				reshape_and_save(source_path, output_file)
				i += 1
				print(f'Reshaped {i} images out of 23554', end='\r')
	

reshape_dataset('emotic', 'emotic_clean_cropped')
