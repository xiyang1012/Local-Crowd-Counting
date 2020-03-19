import os
import sys
import cv2
from scipy.io import loadmat
import numpy as np
import pandas as pd

sys.path.append('../')
from get_density_map_gaussian import get_density_map_gaussian

dataset = ['train', 'test'] # train / test
maxSize = 1024 # (w, h)
minSize = 384  # (w, h)

data_path = '../../ProcessedData/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/'
output_path = '../../ProcessedData/shanghaitech_part_A_mod64/'
if not os.path.exists(output_path):
	os.mkdir(output_path)

if 'test' in dataset:
	# test set
	path = data_path + 'test_data/'
	output_path += 'test/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)

	for idx in range(1, 183):
	# for idx in range(1, 10):
		i = idx
		print("idx: ", i)

		# load gt
		input_img_gt = path + 'ground_truth/' + 'GT_IMG_'+str(idx)+'.mat'
		label = loadmat(input_img_gt)
		annPoints = label['image_info'][0][0][0][0][0]  #（w, h)
		# print(annPoints)
		print('gt sum:', label['image_info'][0][0][0][0][1][0][0])

		input_img_name = path + 'images/' + 'IMG_' + str(i) + '.jpg'
		img = cv2.imread(input_img_name)
		[h, w, c] = img.shape

		# resize
		w_new = (int(w/64)) * 64
		if w_new > 1024:
			w_new = 1024
		elif w_new < 384:
			w_new = 384
		rate_w = float(w_new) / w

		h_new = (int(h/64)) * 64
		if h_new > 1024:
			h_new = 1024
		elif h_new < 384:
			h_new = 384
		rate_h = float(h_new) / h

		img = cv2.resize(img, (w_new, h_new) )
		annPoints[:,0] = annPoints[:,0] * float(rate_w)
		annPoints[:,1] = annPoints[:,1] * float(rate_h)

		# generation
		im_density = get_density_map_gaussian(img, annPoints, 15, 4)
		print('den sum: ', im_density.sum(axis=(0, 1)))

		# save img
		cv2.imwrite(path_img + str(i)+'.jpg', img)

		# save csv
		csv_path = path_den + str(i) + '.csv'
		data_den = pd.DataFrame(im_density)
		data_den.to_csv(csv_path, header=False, index=False)

if dataset == 'train':
	# train set
	path = data_path + 'train_data/'
	output_path += 'train/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)

	for idx in range(1, 301):
	# for idx in range(1, 10):
		i = idx
		print("idx: ", i)

		# load gt
		input_img_gt = path + 'ground_truth/' + 'GT_IMG_' + str(idx) + '.mat'
		label = loadmat(input_img_gt)
		annPoints = label['image_info'][0][0][0][0][0]  # （w, h)
		print('gt sum:', label['image_info'][0][0][0][0][1][0][0])

		input_img_name = path + 'images/' + 'IMG_' + str(i) + '.jpg'
		img = cv2.imread(input_img_name)
		[h, w, c] = img.shape

		# resize
		w_new = (int(w/64)) * 64
		if w_new > 1024:
			w_new = 1024
		elif w_new < 384:
			w_new = 384
		rate_w = float(w_new) / w

		h_new = (int(h/64)) * 64
		if h_new > 1024:
			h_new = 1024
		elif h_new < 384:
			h_new = 384
		rate_h = float(h_new) / h

		img = cv2.resize(img, (w_new, h_new))
		annPoints[:, 0] = annPoints[:, 0] * float(rate_w)
		annPoints[:, 1] = annPoints[:, 1] * float(rate_h)

		# generation
		im_density = get_density_map_gaussian(img, annPoints, 15, 4)
		print('den sum: ', im_density.sum(axis=(0, 1)))

		# save img
		cv2.imwrite(path_img + str(i) + '.jpg', img)

		# save csv
		csv_path = path_den + str(i) + '.csv'
		data_den = pd.DataFrame(im_density)
		data_den.to_csv(csv_path, header=False, index=False)
