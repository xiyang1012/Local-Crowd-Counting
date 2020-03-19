import os
import sys
import cv2
from scipy.io import loadmat
import numpy as np
import pandas as pd

sys.path.append('../')
from get_density_map_gaussian import get_density_map_gaussian

dataset = ['train', 'test'] # train / test
maxSize = 1920 # (w, h)
minSize = 512  # (w, h)

data_path = '../../ProcessedData/UCF-QNRF_ECCV18/'
output_path = '../../ProcessedData/UCF_QNRF_mod64/'
if not os.path.exists(output_path):
	os.mkdir(output_path)

if 'test' in dataset:
	# test set
	path = data_path + 'Test/'
	output_path += 'test/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)

	for idx in range(1, 335):
	# for idx in range(1, 10):
		i = idx
		print("idx: ", i)

		# load gt
		input_img_gt = path + 'img_'+str(idx).zfill(4) + '_ann.mat'
		label = loadmat(input_img_gt)
		annPoints = label['annPoints']  #（w, h)
		# print(annPoints)
		print('gt sum:', annPoints.shape[0])

		input_img_name = path + 'img_' + str(i).zfill(4) + '.jpg'
		img = cv2.imread(input_img_name)
		[h, w, c] = img.shape

		# too large
		if w>maxSize or h>maxSize:
			rate = maxSize / h
			rate_w = w * rate
			if rate_w > maxSize:
				rate = maxSize / w

			w_new = int(w * rate / 64) * 64
			rate_w = float(w_new) / w
			w_new = min(maxSize, max(minSize, w_new))

			h_new = int(h * rate / 64) * 64
			rate_h = float(h_new) / h
			h_new = min(maxSize, max(minSize, h_new))

		# too small
		elif w<minSize or h<minSize:
			rate = minSize / h
			rate_w = w * rate
			if rate_w < minSize:
				rate = minSize / w

			w_new = int(w * rate / 64) * 64
			rate_w = float(w_new) / w
			w_new = min(maxSize, max(minSize, w_new))

			h_new = int(h * rate / 64) * 64
			rate_h = float(h_new) / h
			h_new = min(maxSize, max(minSize, h_new))

		# mod64
		else:
			w_new = (int(w/64)) * 64
			rate_w = float(w_new) / w
			w_new = min(maxSize, max(minSize, w_new))

			h_new = (int(h/64)) * 64
			rate_h = float(h_new) / h
			h_new = min(maxSize, max(minSize, h_new))

		# resize
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
	path = data_path + 'Train/'
	output_path += 'train/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)

	for idx in range(1, 1202):
	# for idx in range(1, 10):
		i = idx
		print("idx: ", i)

		# load gt
		input_img_gt = path + 'img_' + str(idx).zfill(4) + '_ann.mat'
		label = loadmat(input_img_gt)
		annPoints = label['annPoints']  # （w, h)
		# print(annPoints)
		print('gt sum:', annPoints.shape[0])

		input_img_name = path + 'img_' + str(i).zfill(4) + '.jpg'
		img = cv2.imread(input_img_name)
		[h, w, c] = img.shape

		# too large
		if w > maxSize or h > maxSize:
			rate = maxSize / h
			rate_w = w * rate
			if rate_w > maxSize:
				rate = maxSize / w

			w_new = int(w * rate / 64) * 64
			rate_w = float(w_new) / w
			w_new = min(maxSize, max(minSize, w_new))

			h_new = int(h * rate / 64) * 64
			rate_h = float(h_new) / h
			h_new = min(maxSize, max(minSize, h_new))

		# too small
		elif w < minSize or h < minSize:
			rate = minSize / h
			rate_w = w * rate
			if rate_w < minSize:
				rate = minSize / w

			w_new = int(w * rate / 64) * 64
			rate_w = float(w_new) / w
			w_new = min(maxSize, max(minSize, w_new))

			h_new = int(h * rate / 64) * 64
			rate_h = float(h_new) / h
			h_new = min(maxSize, max(minSize, h_new))

		# mod64
		else:
			w_new = (int(w / 64)) * 64
			rate_w = float(w_new) / w
			w_new = min(maxSize, max(minSize, w_new))

			h_new = (int(h / 64)) * 64
			rate_h = float(h_new) / h
			h_new = min(maxSize, max(minSize, h_new))

		# resize
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
