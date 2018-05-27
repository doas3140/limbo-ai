import numpy as np
import time
import tensorflow as tf
import os
from tqdm import tqdm
from models.SL_model import SL_network

def main():
    ep_start = 0
    episodes = 1000
    levels = ['EXPERT']
    learning_rates = [1e-3]
    batch_sizes = [512]
    memory_sizes = [3]
    
    init_model_path = None # None if start from scratch
    # init_model_path = os.path.join( LOGDIR,'EXPERT_0.0001_512_3','saved_model','model.ckpt')
    
    datasets_path = 'E:\\Datasets\\Limbo\\SL_dataset\\'
    # datasets_path = 'C:\\Users\\domin\\Desktop\\sl_network_training\\datasets\\'

    LOGDIR = os.path.join( os.getcwd(),'logdir','SL' )
    for level in levels:
        npy_path = datasets_path+str(level)+'.npy'
        D = np.load(npy_path)
        D_len = D.shape[0]
        for learning_rate in tqdm(learning_rates,desc='learn_rate'):
            for batch_size in tqdm(batch_sizes,desc='batch_size'):
                for memory_size in tqdm(memory_sizes,desc='mem_size__'):
                    logdir = os.path.join( LOGDIR, str(level)+'_'+str(learning_rate)+'_'+str(batch_size)+'_'+str(memory_size) )
                    tf.logging.set_verbosity(tf.logging.ERROR)
                    tf.reset_default_graph()
                    nn = SL_network(LOGDIR=logdir,learning_rate=learning_rate, memory_size=memory_size,init_model_path=init_model_path)
                    for episode in tqdm(range(ep_start+1,ep_start+episodes+1),desc='episode___'):
                        for iteration in tqdm(range(int(D_len/batch_size)),desc='iteration_'):
                            s_batch,a_batch,v_batch = get_random_batch(D, batch_size, memory_size)
                            nn.update(s_batch,a_batch,v_batch)
                            if iteration%int(100*(np.max(batch_sizes)/batch_size)) == 0:
                                nn.save_summary(s_batch,a_batch,v_batch)
                        if episode%1 == 0:
                            nn.save_model()

def get_random_batch(D,batch_size, memory_size=5):
    indexes = np.random.choice(np.arange(memory_size,len(D)),batch_size)
    s_batch = []
    a_batch = []
    v_batch = []
    for i in indexes:
        s_temp = []
        for m in range(memory_size)[::-1]: # 4 -> 0
            s_temp.append(D[i-m][0][0])
        s_batch.append(np.array(s_temp)) # (1,261)->(m,261)
        a_batch.append(D[i][1]) # (10,)
        v_batch.append(D[i][2]) # ()
    v_batch = np.reshape(v_batch,[batch_size,1])
    v_batch = (v_batch+600)/(600+500) # v_batch is normalized [0-1]
    return np.array(s_batch), np.array(a_batch), np.array(v_batch)

if __name__ == '__main__':
    main()