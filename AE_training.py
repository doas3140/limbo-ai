import numpy as np
import time
import tensorflow as tf
import os
from models.AE_model import AutoEncoder
from tqdm import trange,tqdm

def main():
    dataset_dir = 'E:/Datasets/Limbo/autoencoder_data/'
    dataset_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

    LOGDIR = os.path.join(os.getcwd(),'logdir','AE')
    learning_rates = [1e-4]
    batch_sizes = [64]
    init_model_path = None
    # init_model_path = os.getcwd()+'\\logdir\\AE\\0.0001_64\\saved_model\\model.ckpt'
    episodes = 20000

    for learning_rate in tqdm(learning_rates, desc='learning_rate'):
        for batch_size in tqdm(batch_sizes, desc='batch_size'):
            tf.logging.set_verbosity(tf.logging.ERROR) # filter INFO logs
            tf.reset_default_graph() # reset graph
            logdir = os.path.join(LOGDIR,str(learning_rate)+'_'+str(batch_size))
            nn = AutoEncoder(logdir,learning_rate,init_model_path=init_model_path)
            for counter in trange(episodes):
                D = get_random_file(dataset_dir, dataset_files)
                batches = []
                for _ in range(5):
                    batch = get_random_batch(D,batch_size)
                    batches.append(batch)
                for img_batch,feat_batch in batches:
                    nn.update(feat_batch,img_batch)
                    if counter%10 == 0:
                        nn.save_summary(feat_batch,img_batch)
                    if counter%1000 == 0:
                        nn.save_model()

def get_random_file(dataset_dir, dataset_files):
    random_file = np.random.choice(dataset_files)
    return np.load(dataset_dir+random_file)
    
def get_random_batch(D,batch_size):
    indexes = np.random.choice(len(D),batch_size)
    img_batch = []
    feat_batch = []
    for i in indexes:
        img_batch.append(D[i][0])
        feat_batch.append(D[i][1][0]) # (1,5,8,1280)
    img_batch = np.reshape(img_batch,[-1,144,256,1])/255
    return np.array(img_batch), np.array(feat_batch)

if '__main__' == __name__:
    main()