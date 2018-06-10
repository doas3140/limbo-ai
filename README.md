# Limbo AI  
Tensorflow implementation of Artificial Intelligence in game [Limbo](https://store.steampowered.com/app/48000/LIMBO/).
 Works only on Windows.  
![alt text](https://github.com/doas3140/limbo-ai/raw/master/gifs/0.gif "SL output")  


## Architecture

Consists of 4 neural networks: AutoEncoder (AE), Object Detection (OD), Supervised Learning (SL) and Reinforcement Learning(RL).  
You can try to collect data and train models yourself. Or you can download pretrained models (see below). Reinforcement Learning part doesn't work now (converges to one action too quickly or doesn't learn at all).  
![alt text](https://github.com/doas3140/limbo-ai/raw/master/gifs/architecture.png "Architecture")  

## Collecting data

 1) In `collect_data/collect.py` change `DATASET_FOLDER` to output folder. Also change `SCREEN_REGION` to match your resolution.
 2) If you're playing in windowed mode you can find coordinates of screen with `python utils/show_coords.py`.
 3) Start script using `python collect_data/collect.py`. To pause/unpause recording hit `P` key.
 4) This will create npy files consisting of list of <frame,action,reward>. Rewards are created using Optical Flow. Moving right leads to positive rewards, left - negative.
 5) To extract images from npy files run `python utils/npy2images.py`. Change `NPY_FOLDER_PATH` to path of recorded npy frames and `OUTPUT_FOLDER_PATH` to output path.

## Object Detection network training

Check `OD_training` folder.  
   
## AutoEncoder network training
 
 1) In `AE_create_dataset.py` change `INPUT_DIR` to path to your collected dataset folder. And `OUTPUT_DIR` to output folder.
 2) Create dataset for Autoencoder using `python AE_create_dataset.py`.
 3) In `AE_training.py` change `dataset_dir` to path to your dataset (got from 2nd step).
 4) Start training by `python AE_training.py`.
 5) To check progress access tensorboad by `tensorboard --logdir path_to_this_repo/logdir/AE` 

## Supervised Learning network training

 1) In `SL_create_dataset.py` change `DATASET_PATH` to path to dataset. Dataset must be in particular format.
 Root folder must contain `EXPERT` and `BASIC` folders. In each folder there should be folders of games (one folder = one game).
 Each game folder must contain files named `D0.npy,D1.npy,...`. Also change `OUTPUT_DATASET_PATH` to output directory.
 2) In `SL_create_dataset.py` change `OD_init` to full path to `logdir/OD` if needed. Also change `AE_init`
 to `logdir/AE/{any_params}/saved_model/model.ckpt` if needed.
 3) Create dataset using `python SL_create_dataset.py`
 4) In `SL_training.py` change `datasets_path` to dataset path containing `EXPERT.npy` and `BASIC.npy` (output of 3rd step).
 To continue training change `init_model_path` to path of saved model.

## Reinforcement Learning network training

 `DOESNT WORK NOW` (converges to one action too quickly or doesn't learn at all)
 
## Using pretrained models

 1) Download zip from [logdir.zip](https://drive.google.com/file/d/1xJ3El1DqX1h9LAgBvjtPs0mHHnOpG7Tv/view?usp=sharing).
 2) Replace `logdir` folder with folder inside downloaded zip file.
 
## Testing SL or RL network in Limbo environment

 1) In `test_agent.py` change `OD_init` to frozen_graph folder `logdir/OD`. Also change `AE_init` and `SL_init` to paths to saved models (to `model.ckpt`).
 2) Change `SCREEN_REGION` (frame that is fed through networks) and `RECORD_REGION` (frame that is recorded for gif images) to match your needs.
 3) Change other parameters to match previous trained networks (if params in previous models were changed).
 4) To watch value-function in real-time run `python liveplot/show.py`.
 5) Run script `python test_agent.py`. To pause/unpause recording hit `P` key.