import torch
import torch.onnx
import onnx
import cv2

from collections import OrderedDict
import os
import pickle 

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, LazyConfig
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from detectron2.export.flatten import TracingAdapter

from torchvision import transforms
from PIL import Image


# set these to match your model
model_name = "R-50_RGB_60k"
weights_path = "models/PercepTreeV1/R-50_RGB_60k.pth"
detectron2_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

export_path = "exports/PercepTreeV1/" + model_name + "/"


if __name__ == "__main__":
    torch.cuda.is_available()
    logger = setup_logger(name=__name__)
    
    # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py        
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file(detectron2_config_file))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.MODEL.MASK_ON = True
    
    cfg.OUTPUT_DIR = './output'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time


    #export config file
    with open(export_path + model_name + ".yaml", "w") as f:
        f.write(cfg.dump())


    #load model
    model = build_model(cfg)
    model.eval()
    model.backbone.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(weights_path) 


    #export weights to pkl file
    weights = OrderedDict()

    for p in checkpointer.model.state_dict():
        weights[p] = checkpointer.model.state_dict()[p].cpu().numpy()

    with open(os.path.join(export_path + model_name + ".pkl"), 'wb') as f:
        myModel = {'model': weights, '__author__': "detectron2_converter"}
        pickle.dump(myModel, f)
