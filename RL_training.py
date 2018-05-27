from models.OD_model import OD_network
from models.AE_model import AutoEncoder as AE_network
from models.RL_model import RL_network
from environment.Environment import Environment
import time
import cv2
import numpy as np
import os
from tqdm import tqdm
from collections import deque
from environment.keycheck import key_check
import sys
import imageio
from environment.grabscreen import grab_screen
from liveplot.liveplot import LivePlot

def main():
    SUMM_FOLDER_NAME = 'original'
    PLOT_LEN = 500
    PLOT_N = 3 # skip N frames
    MEMORY_SIZE = 3
    SCREEN_REGION = (3,33,1024,606)
    BATCHES = 64
    EPISODE_LENGTH = 20
    REPLAY_MEMORY_SIZE = 100 # total=this*episode_length
    try:
        plot = LivePlot(MAXLEN=PLOT_LEN,N=PLOT_N)
        OD_init = os.path.join( os.getcwd(),'logdir','OD')
        AE_init = os.path.join( os.getcwd(),'logdir','AE','0.0001_64','saved_model','model.ckpt' ) 
        RL_init = None
        RL_LOGDIR = os.path.join( os.getcwd(),'logdir','RL',SUMM_FOLDER_NAME )
        OD = OD_network(init_folder_path=OD_init)
        AE = AE_network(init_model_path=AE_init,testing=True)
        RL = RL_network(LOGDIR=RL_LOGDIR,memory_size=MEMORY_SIZE,init_model_path=RL_init)
        env = Environment(SCREEN_REGION)
        REPLAY_MEMORY = deque(maxlen=REPLAY_MEMORY_SIZE)

        while True:
            new_frame,done = env.reset()
            state_memory = deque(maxlen=MEMORY_SIZE)
            r_array = [] # (n,)
            v_array = [] # (n,)
            s_array = [] # (n,[5,261])
            a_array = [] # (n,[10,])
            start_time = time.time()
            for counter in range(EPISODE_LENGTH+MEMORY_SIZE+1): # +1 for last v
                # calculate current s
                objects, features = OD.forward(new_frame)
                encoded_features = AE.forward(features)
                s = concat_outputs(objects, encoded_features, OD) # (1,261)
                s = s[0] # (261)
                state_memory.append(s)
                # calc next a
                if len(state_memory) > MEMORY_SIZE-1:
                    s_input = np.expand_dims(np.array(state_memory), axis=0) # (1,m,261)
                    a_probs, v_normalized = RL.forward(s_input) # (1,nA) (1,1)
                    v_normalized = v_normalized[0][0] # (1,1) -> ()
                    v = v_normalized*(500+600)-600
                    plot.emit(v)
                else:
                    a_probs = np.array([[0,0,0,0,0,0,0,0,0,1]])
                a = probs_to_onehot_action(a_probs)
                # take step
                new_frame,r,done = env.step(a)
                # save
                if len(state_memory) > MEMORY_SIZE-1:
                    r_array.append(r)
                    v_array.append(np.expand_dims(v_normalized,axis=0))
                    s_array.append(state_memory)
                    a_array.append(a)
                # check if done
                if done:
                    break
                # print time
                total_time = time.time()-start_time
                a_probs = np.array2string(a_probs, formatter={'float_kind':lambda x: "%.2f" % x})
                print('time:{:>20} frame:{:>6} a:{}'.format(total_time,counter,a_probs),end='\r')
            if len(r_array) < 10:
                continue # if insta done then skip to next frame
            print('') # when finished
            env.stop()
            if len(r_array) < EPISODE_LENGTH:
                r_array[-2] = -50
            v_correct = calc_v(r_array,v_end=v_array[-1])
            # create advantage array
            adv_array = calc_advantage(v_correct,np.array(v_array[:-1]))
            # shuffle arrays
            random_index = np.arange(len(v_array[:-1]))
            np.random.shuffle(random_index)
            adv_array = np.array(adv_array)[random_index]
            v_array = np.array(v_array[:-1])[random_index]
            s_array = np.array(s_array[:-1])[random_index]
            a_array = np.array(a_array[:-1])[random_index]
            # split into baches
            adv_array = split(adv_array,BATCHES)
            v_array = split(v_array,BATCHES)
            s_array = split(s_array,BATCHES)
            a_array = split(a_array,BATCHES)
            # update weights
            for i in range(len(a_array)):
                REPLAY_MEMORY.append([s_array[i],a_array[i],v_array[i],adv_array[i]])

            indexes = np.random.choice(np.arange(len(REPLAY_MEMORY)),len(REPLAY_MEMORY))
            for i in indexes:
                s_array,a_array,v_array,adv_array = REPLAY_MEMORY[i]
                RL.update(s_array,a_array,v_array,adv_array)
                print('updating weights {}/{} adv_mean: {}'.format(i,len(REPLAY_MEMORY),np.mean(adv_array)),end='\r')
            print('')

    except KeyboardInterrupt:
        env.stop()
        sys.exit()

def split(arr,batch_size):
    res = []
    temp = []
    for i in range(len(arr)):
        temp.append(arr[i])
        if len(temp) > batch_size-1:
            res.append(np.array(temp))
            temp = []
    if len(temp) > 0:
        res.append(np.array(temp))
    return np.array(res)

def calc_v(r_array,v_end,y=0.998):
    v_decayed = []
    v = v_end*(500+600)-600 # denormalize
    for r in r_array[:-1][::-1]: # till last + reversed
        v = r + y*v
        v_decayed.append(v)
    return (np.array(v_decayed)+600)/(500+600) # normalize

def calc_advantage(v_correct,v_predicted): # (n,) (n,)
    v_correct = v_correct*(500+600)-600
    v_predicted = v_predicted*(500+600)-600
    return (v_correct - v_predicted)+600/(500+600)
        

def concat_outputs(objects,encoded_features, OD): # (n,7,19), (1,128)
    objects = np.reshape(objects,[1,OD.NUM_DETECTIONS*(4+1+OD.NUM_CLASSES)]) # 7*(boxes+score+14)
    return np.concatenate([encoded_features,objects],axis=1) # (1,261)

def probs_to_onehot_action(a_probs): # (1,nA)
    a = np.random.choice(a_probs.shape[1], p=a_probs[0,:])
    onehot = np.zeros([a_probs.shape[1]])
    onehot[a] = 1
    return onehot

if __name__ == '__main__':
    main()