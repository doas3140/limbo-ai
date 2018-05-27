from models.OD_model import OD_network as OD_network
from models.AE_model import AutoEncoder as AE_network
import numpy as np
import os
from tqdm import tqdm
from collections import deque

def main():
    ''' !!! DATASET_PATH should contain folders 'EXPERT','BASIC'              !!!
        !!! in each folder should be folders of games (one folder = one game) !!! 
        !!! one game folder must contain files named D0.npy,D1.npy,...        !!! '''

    DATASET_PATH = 'E:/Datasets/Limbo/SL'
    OUTPUT_DATASET_PATH = 'E:/Datasets/Limbo/SL_dataset'

    OD_init = os.path.join( os.getcwd(),'logdir','OD')
    AE_init = os.path.join( os.getcwd(),'logdir','AE','0.0001_64','saved_model','model.ckpt' ) 
    OD = OD_network(init_folder_path=OD_init)
    AE = AE_network(init_model_path=AE_init,testing=True)

    GAMMA = 0.998

    for folder in tqdm(['BASIC','EXPERT'],desc='levels_'):
        main_folder_path = os.path.join(DATASET_PATH,folder)
        folders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
        OUTPUT = []
        for folder_num in tqdm(folders,desc='folders'):
            game_folder_path = os.path.join(main_folder_path,folder_num)
            num_files = len(os.listdir(game_folder_path))
            v = 0 # init v
            for file_nr in tqdm(range(0,num_files)[::-1],desc='files__'):
                npy_path = os.path.join(game_folder_path,'D'+str(file_nr)+'.npy')
                D = np.load(npy_path)
                for i in tqdm(range(len(D))[::-1],desc='frames_'):
                    s = D[i][0] # (144,256)
                    a = D[i][1] # (10,)
                    r = D[i][2] # ()
                    if s.max() < 100: # r=-2 if frame is black(dead)
                        r = -2
                    v = r + GAMMA*v # update v
                    objects, features = OD.forward(s)
                    encoded_features = AE.forward(features)
                    s = concat_outputs(objects, encoded_features, OD)
                    OUTPUT.append([s,a,v])
        save_dir = os.path.join(OUTPUT_DATASET_PATH,folder+'.npy')
        np.save(save_dir, OUTPUT)

def concat_outputs(objects,encoded_features, OD): # (n,7,19), (1,128)
    objects = np.reshape(objects,[1,OD.NUM_DETECTIONS*(4+1+OD.NUM_CLASSES)]) # 7*(boxes+score+14)
    return np.concatenate([encoded_features,objects],axis=1) # (1,261)

if __name__ == '__main__':
    main()