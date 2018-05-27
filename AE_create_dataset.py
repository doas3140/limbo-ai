import os
import numpy as np
import tensorflow as tf
import sys
from tqdm import trange,tqdm
from models.OD_model import OD_network

OD_init = os.path.join( os.getcwd(),'logdir','OD' )
nn = OD_network(OD_init)

INPUT_DIR = 'E:/Datasets/Limbo/AE_create/'
OUTPUT_DIR = 'E:/Datasets/Limbo/autoencoder_data/'

data = []
counter = 0

INPUT_FILES = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
for input_file in tqdm(INPUT_FILES,desc='files'):
    D = np.load(os.path.join(INPUT_DIR,input_file))
    for i in trange(0,len(D)):
        frame = D[i][0] # (144,256)
        _,features = nn.forward(frame)
        if len(data) > 1000-1:
            np.save(OUTPUT_DIR+'D'+str(counter)+'.npy',data)
            data = []
            counter += 1
        
        data.append(np.array([ frame,features ]))