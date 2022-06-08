import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from models.experimental import attempt_load
from utils.general import non_max_suppression, set_logging
from utils.torch_utils import select_device

weights = './runs/train/yolov5l_result/weights/best.pt'

imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
device = ''
classes = None
agnostic_nms = False
half = False
hide_conf = True
thickness = 2
rect_thickness = 3
pred_shape = (480, 640, 3)
vis_shape = (800, 600)

cap = cv2.VideoCapture(0)

set_logging()
device = select_device(device)
half &= device.type != 'cpu' 
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

while 1:
    ret, frame = cap.read()
    out = frame.copy()
    frame = cv2.resize(frame, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_LINEAR)
    frame = np.transpose(frame, (2, 1, 0))

    cudnn.benchmark = True  # set True to speed up constant image size inference

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    frame = torch.from_numpy(frame).to(device)
    frame = frame.float()
    frame /= 255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)

    frame = torch.transpose(frame, 2, 3)

    pred = model(frame, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # detections per image
    for i, det in enumerate(pred):
        
        img_shape = frame.shape[2:]
        out_shape = out.shape

        s_ = f'{i}: '
        s_ += '%gx%g ' % img_shape  # print string

        if len(det):

            coords = det[:, :4]

            gain = min(img_shape[0] / out_shape[0], img_shape[1] / out_shape[1])  # gain  = old / new
            pad = (img_shape[1] - out_shape[1] * gain) / 2, (
                    img_shape[0] - out_shape[0] * gain) / 2  # wh padding

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain

            coords[:, 0].clamp_(0, out_shape[1])  # x1
            coords[:, 1].clamp_(0, out_shape[0])  # y1
            coords[:, 2].clamp_(0, out_shape[1])  # x2
            coords[:, 3].clamp_(0, out_shape[0])  # y2

            det[:, :4] = coords.round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s_ += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
            for *xyxy, conf, cls in reversed(det):
        
                c = int(cls)  # integer class
                label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'

                tl = rect_thickness

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(out, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(out, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(out, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)

        print(f'{s_}Done.')
        
    out = cv2.resize(out, vis_shape, cv2.INTER_LINEAR)
    cv2.imshow("out", out)
    
    if cv2.waitKey(5) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break