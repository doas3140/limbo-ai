import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from models.utils import label_map_util
from models.utils import visualization_utils as vis_util

class OD_network():
    def __init__(self,init_folder_path,num_classes=14,num_detections=7,t=0.3):
        self.LOGDIR = init_folder_path
        self.NUM_CLASSES = num_classes
        self.THRESHOLD =  t
        self.NUM_DETECTIONS = num_detections

        PATH_TO_CKPT = os.path.join(self.LOGDIR,'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(self.LOGDIR,'labelmap.pbtxt')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.feature_tensor = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV2/Conv_1/Relu6:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def forward(self,frame): # (144,256)
        arr = np.array([ frame for i in range(3) ]) # (3,144,256)
        frame = np.rollaxis(arr, 0, 3) # (144,256,3)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame_expanded = np.expand_dims(frame, axis=0) # (1,144,256,3)
        
        [boxes, scores, classes, num, features] = self.sess.run(
            [self.detection_boxes, self.detection_scores, 
            self.detection_classes, self.num_detections, self.feature_tensor],
            feed_dict={self.image_tensor: frame_expanded})
        
        total_elem = []

        for i in range(100):
            if scores[0][i] > self.THRESHOLD:
                e = np.zeros([4+1+self.NUM_CLASSES])
                e[0] = boxes[0][i][0]
                e[1] = boxes[0][i][1]
                e[2] = boxes[0][i][2]
                e[3] = boxes[0][i][3]
                e[4] = scores[0][i] # prob of current class
                # one-hot enc of class
                # !!! classes are from 1 to num_classes
                one_hot = [0]*self.NUM_CLASSES
                one_hot[ int(classes[0][i])-1 ] = 1
                for j,o in enumerate(one_hot):
                    e[j+5] = o
                total_elem.append(e)
        
        if len(total_elem) > self.NUM_DETECTIONS:
            total_elem = total_elem[:self.NUM_DETECTIONS]
        elif len(total_elem) < self.NUM_DETECTIONS:
            num_missing = self.NUM_DETECTIONS - len(total_elem)
            missing_boxes = np.zeros([19])
            for i in range(num_missing):
                total_elem.append(np.zeros(4+1+self.NUM_CLASSES))
                
        return np.array(total_elem), features

    def forward_vis_frame(self,frame): # (144,256)
        ''' output is frame with object detection for visual purpose '''
        arr = np.array([ frame for i in range(3) ]) # (3,144,256)
        frame = np.rollaxis(arr, 0, 3) # (144,256,3)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame_expanded = np.expand_dims(frame, axis=0) # (1,144,256,3)
        
        [boxes, scores, classes, num, features] = self.sess.run(
            [self.detection_boxes, self.detection_scores, 
            self.detection_classes, self.num_detections, self.feature_tensor],
            feed_dict={self.image_tensor: frame_expanded})
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=self.THRESHOLD)
        
        return frame, features