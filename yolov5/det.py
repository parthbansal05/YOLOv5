from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path
import time

import torch
import torch.backends.cudnn as cudnn

from utils.augmentations import letterbox
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
def detect(
    weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
):

    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    device = ''
    classes = None
    agnostic_nms = False

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    
    im0 = cv2.imread(source)  # BGR

    im = letterbox(im0, 640, stride, pt)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    # Inference
    pred = model(im, augment=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    lables=[]
    cords=[]

    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                lables.append(xywh)
                cords.append(label)
                # LOGGER.info(str(xywh) + '\n')
                # LOGGER.info(str(label) + '\n')
    return lables, cords
