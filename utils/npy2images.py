import numpy as np
import cv2
import os
from tqdm import trange,tqdm

NPY_FOLDER_PATH = 'E:/Datasets/Limbo/game_1'
OUTPUT_FOLDER_PATH = 'E:/Datasets/Limbo/output'
K = 5 # only every K frame is saved

npy_files = [f for f in os.listdir(NPY_FOLDER_PATH) if os.path.isfile(os.path.join(NPY_FOLDER_PATH, f))]
file_counter = 0
k_counter = 0
for npy_file in tqdm(npy_files):
	npy_file_path = os.path.join( NPY_FOLDER_PATH, npy_file )
	D = np.load(npy_file_path)
	for i in trange(len(D)):
		k_counter += 1
		if K < k_counter:
			k_counter = 0
			frame = D[i][0]
			output_path = os.path.join(OUTPUT_FOLDER_PATH,str(file_counter)+'.png')
			cv2.imwrite(output_path,frame)
			file_counter += 1