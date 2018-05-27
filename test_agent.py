from models.OD_model import OD_network
from models.AE_model import AutoEncoder as AE_network
from models.SL_model import SL_network
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
    RECORD_GIF = True
    V_LEN = 500 # record last 500 frames
    V_N = 3 # record v of every n'th frame
    MEMORY_SIZE = 3
    SCREEN_REGION = (3,33,1024,606)
    # RECORD_REGION = (3,33,1024,606) # record for gif
    RECORD_REGION = (3,33,1624,606)
    try:
        OD_init = os.path.join( os.getcwd(),'logdir','OD')
        AE_init = os.path.join( os.getcwd(),'logdir','AE','0.0001_64','saved_model','model.ckpt' ) 
        SL_init = os.path.join( os.getcwd(),'logdir','SL','EXPERT_0.001_512_3','saved_model','model.ckpt' ) 
        OD = OD_network(init_folder_path=OD_init)
        AE = AE_network(init_model_path=AE_init,testing=True)
        SL = SL_network(init_model_path=SL_init,testing=True,memory_size=MEMORY_SIZE)

        GIF = []
        GIF_N = 4 # every N frames record
        GIF_K = 3 # times lower resolution
        GIF_counter = 0

        plot = LivePlot(MAXLEN=V_LEN,N=V_N)
        
        state_memory = deque(maxlen=MEMORY_SIZE)
        env = Environment(SCREEN_REGION)
        new_frame,done = env.reset()
        paused = True
        while True:
            if not paused:
                GIF_counter += 1
                start_time = time.time()
                if RECORD_GIF:
                    if GIF_counter > GIF_N-1:
                        GIF_counter = 0
                        r_screen = grab_screen(RECORD_REGION)
                        new_h,new_w = int(r_screen.shape[0]/GIF_K), int(r_screen.shape[1]/GIF_K)
                        GIF.append(cv2.resize(r_screen, (new_w,new_h)))
                        # GIF.append(r_screen)
                # run OD and AE nn's
                objects, features = OD.forward(new_frame)
                encoded_features = AE.forward(features)
                s = concat_outputs(objects, encoded_features, OD)
                state_memory.append(s[0])
                if len(state_memory) > MEMORY_SIZE-1:
                    s_input = np.expand_dims(np.array(state_memory), axis=0) # (1,m,261)
                    a_probs, v_normalized = SL.forward(s_input) # (1,nA) (1,1)
                    v = v_normalized*(500+600)-600
                    plot.emit(v[0][0]) # (1,1) -> (,)
                    a = probs_to_onehot_action(a_probs)
                else:
                    a = np.array([0,0,0,0,0,0,0,0,0,1]) # 'nothing' action
                new_frame,r,done = env.step(a)
                if len(state_memory) > MEMORY_SIZE-1:
                    total_time = time.time()-start_time
                    a_probs = np.array2string(a_probs, formatter={'float_kind':lambda x: "%.2f" % x})
                    print('time:{:>20} a:{}'.format(total_time,a_probs),end='\r')
            else:
                time.sleep(0.3)
                print('paused')
            paused,GIF = check_pause(paused,GIF,RECORD_GIF)
    except KeyboardInterrupt:
        env.stop()
        sys.exit()

def concat_outputs(objects,encoded_features, OD): # (n,7,19), (1,128)
    objects = np.reshape(objects,[1,OD.NUM_DETECTIONS*(4+1+OD.NUM_CLASSES)]) # 7*(boxes+score+14)
    return np.concatenate([encoded_features,objects],axis=1) # (1,261)

def check_pause(paused,GIF,RECORD_GIF):
    pressed_keys = key_check()
    if 'P' in pressed_keys or 'p' in pressed_keys:
        if paused:
            print('STARTED')
            time.sleep(0.3)
            GIF = []
            return False,GIF
        else:
            print('PAUSED')
            time.sleep(0.3)
            if RECORD_GIF:
                save_gif(GIF)
            return True,GIF
    else:
        return paused,GIF

def probs_to_onehot_action(a_probs):
    a = np.random.choice(a_probs.shape[1], p=a_probs[0,:])
    onehot = np.zeros([a_probs.shape[1]])
    onehot[a] = 1
    return onehot

def save_gif(GIF):
    DATASET_FOLDER = os.path.join(os.getcwd(),'gifs')
    num_gifs = len([f for f in os.listdir(DATASET_FOLDER) if os.path.isfile(os.path.join(DATASET_FOLDER, f))])
    imageio.mimsave(os.path.join(DATASET_FOLDER,str(num_gifs)+'.gif'), GIF)

if __name__ == '__main__':
    main()