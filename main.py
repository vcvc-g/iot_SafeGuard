from time import sleep
import RPi.GPIO as gp
import board
import adafruit_mlx90614
import busio

import cv2
import argparse
import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from PIL import Image, ImageDraw, ImageFont
import datetime

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
id2_tip_class = {0: 'Thank you for your corporation', 1: 'Please wear a face mask'}
colors = ((0, 255, 0), (255, 0 , 0))

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
def inference(net, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True):
    
    result = []
    
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net.setInput(blob)
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    tl = round(0.002 * (height + width) * 0.5) + 1  # line thickness
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        
        result.append(class_id)
        
        if draw_result:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
            cv2.putText(image, "%s" % (id2_tip_class[class_id]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
    return image, result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--proto', type=str, default='models/face_mask_detection.prototxt', help='prototxt path')
    parser.add_argument('--model', type=str, default='models/face_mask_detection.caffemodel', help='model path')
    args = parser.parse_args()
    Net = cv2.dnn.readNet(args.model, args.proto)
    conf_thresh=0.5

    #init for themometer
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90614.MLX90614(i2c)
    highest_T = 37.5

    # init for press sensor
    gp.setmode(gp.BCM)
    gp.setup(23, gp.IN, pull_up_down = gp.PUD_DOWN)

    # init for steer
    gp.setup(18, gp.OUT)  
    pwm = gp.PWM(18, 50)  
    pwm.start(0)
    
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
    
    while True:

        #get T from object
        ambient_T = mlx.ambient_temperature
        objtect_T = mlx.object_temperature
        print("ambient_T",ambient_T)
        print("objtect_T",objtect_T)
        if objtect_T - ambient_T < 1 or objtect_T - ambient_T < -1:
            print("No person detected")
            sleep(0.03)
        elif objtect_T < highest_T and objtect_T > 30:
            sleep(0.03)
            
            status = True
            all_masked = False
            while status:
                status, img_raw = cap.read()
                if not status:
                    break
                
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                img_raw, result = inference(Net, img_raw, target_shape=(260, 260), conf_thresh=conf_thresh)
                cv2.imshow('image', img_raw[:,:,::-1])
                cv2.waitKey(1)	
                
                all_masked = True if len(result) == result.count(0) else False
                press_value = gp.input(23)
                
                if not press_value:
                    print("Please use hand sanitizer")
                if not all_masked:
                    print("Please mask up")
                if press_value and all_masked:
                    # fast change angle
                    pwm.ChangeDutyCycle(7.5) #set to 90, open the door
                    sleep(0.1)
                    pwm.ChangeDutyCycle(0)
                    sleep(3) 
                    pwm.ChangeDutyCycle(2.5) #set to 0, close the door
                    sleep(0.1)
                    pwm.ChangeDutyCycle(0)
                    break               
                    
            cv2.destroyAllWindows()

        elif objtect_T < 30 and objtect_T - ambient_T > 1:
            print("Please be closer to themometer")
        else:
            print("Your temperature is too high")
    #     gp.cleanup()
