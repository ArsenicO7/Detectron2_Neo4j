from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np

#json for json file output and os for file paths
import json
import os

class Detector:
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()

        # Load model config and pretrained model
        if model_type == "OD": #Object Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        elif model_type == "IS": #Instance Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "KP": #Keypoint Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "LVIS": #LVIS Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" #cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)
        
    def onImage(self, imagePath, frameNum):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
        instance_mode = ColorMode.SEGMENTATION)

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        # Naming a window
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
  
        # Using resizeWindow()
        cv2.resizeWindow("Result", 1000, 1000)

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)

        #Create new variable with pred_classes label
        pred_classes = predictions['instances'].pred_classes.cpu().tolist()
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
        pred_class_names = list(map(lambda x: class_names[x], pred_classes))

        #prints the arrays of all instances and their elements
        #print ((predictions["instances"]))

        #for loop to go through each instance in a frame
        for x in range (len(predictions["instances"])):
            print(pred_class_names[x])

        #create variables for JSON file utilizing instance info from frame
        boxes = str(predictions['instances'].pred_boxes[0])
        classification = pred_class_names[0]

        #sample data for JSON file for one frame and the instances. Would put in for loop
        data = {
            "frame_number": frameNum,
            "obj_number": len(predictions["instances"]),
            "objects" : classification,
            "box" : boxes
        }

        with open('{}.json'.format(frameNum), 'w') as f:
            json.dump(data, f, indent=2)
            print("New json file is created from data.json file")

    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        
        if (cap.isOpened()==False):
            print ("Error opening file...")
            return
        
        (success, image) = cap.read()

        while success:

            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            # Naming a window
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

            # Using resizeWindow()
            cv2.resizeWindow("Result", 1000, 1000)

            cv2.imshow("Result", output.get_image()[:,:,::-1])

            key=cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()