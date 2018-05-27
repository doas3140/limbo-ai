# Object Detection network

SSD mobilenet v2 architecture training using Tensorflow Object Detection API.

## Preparation

 1) Collect data (see "collecting data" in root folder) or download [OD_training.zip](https://drive.google.com/file/d/1BvPQUHiKSr6S8meybGxYAXLnomoeJVgY/view?usp=sharing)
  and skip 2 step and replace folders from zip file in steps 3-4.
 2) Annotate pictures using ["labelimg"](https://github.com/tzutalin/labelImg).
 3) Put all images in `dataset/images`. Training annotations (xml files) in `dataset/train_ann` and validation annotations in `dataset/valid_ann`.
 4) Download ssd_mobilenet_v2_coco_2018_03_29 folder from Tensorflow model zoo and place here.
 5) If more classes then change /config/labelmap.pbtxt
 6) edit /config/ssd_mobilenet_v2.config: `num_classes`, `resizer` (optional), `batch_size` (optional), `eval_config.num_exaples` (num of validation annotations).  
 Change directories in: `fine_tune_checkpoint`, `train_input_reader.input_path`, `train_input_reader.label_map_path`, `eval_input_reader.input_path`, `eval_input_reader.label_map_path`.

## Training

 1) Convert xml annotations to csv using `python xml_to_csv.py`.
 2) Generate tfrecords using `python generate_tfrecords.py`.
 3) Train model using `python train.py --logtostderr --train_dir=training/ --pipeline_config_path=config/ssd_mobilenet_v2.config`.
 4) To check progress access tensorboad with `tensorboard --logdir path_to_here/training`.

## Freezing model for further use

 1) Run `python export_inference_graph.py --input_type image_tensor --pipeline_config_path config/ssd_mobilenet_v2.config --trained_checkpoint_prefix training/model.ckpt-{episode_number} --output_directory frozen_graph`. Change `{episode_number}` to current number (see `training` folder).
 2) Copy `/config/labelmap.pbtxt` to `/frozen_graph/`.
 3) Copy every file from `OD_training/training/` to `logdir/OD/`.