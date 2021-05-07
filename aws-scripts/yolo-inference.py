from argparse import Namespace
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import base64

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#args
opt = Namespace()
opt.img_size = 640
opt.conf_thres = 0.25
opt.iou_thres = 0.45
opt.device = 'cpu'
opt.agnostic_nms = True
opt.augment = True
opt.classes = None

import logging, requests, os, io, glob, time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

#global vars
device = select_device(opt.device)



def model_fn(model_dir):
    imgsz = opt.img_size

    # Initialize
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    print('load model')
    model = attempt_load(f'{model_dir}/model.pth', map_location=device)  # load FP32 model
    print(model)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    
    return model
    
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    ''' Deserialize the Invoke request body into an object 
    For inference '''
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE:
        jpg_original = base64.b64decode(request_body)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        print('got image')
        return img
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))        

def predict_fn(input_object, model):
    half = device.type != 'cpu'
    
    imgsz = opt.img_size
    img0 = input_object
    stride = int(model.stride.max())  # model stride
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    pred_labels = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = {}
                line['label'] = int(cls.item())
                line['x'] = xywh[0]
                line['y'] = xywh[1]
                line['width'] = xywh[2]
                line['height'] = xywh[3]
                line['conf'] = conf.item()
                pred_labels.append(line)
    
    return pred_labels

# def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
#     logger.info('Serializing the generated output.')
#     if accept == JSON_CONTENT_TYPE:
#         output = json.dumps(prediction)
#         return output, accept
#     raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept)) 


