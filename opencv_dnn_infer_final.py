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
        if draw_result:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
            cv2.putText(image, "%s" % (id2_tip_class[class_id]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
    return image

def run_on_video(Net, video_path, conf_thresh=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    while status:
        status, img_raw = cap.read()
        if not status:
            print("Done processing !!!")
            break
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_raw = inference(Net, img_raw, target_shape=(260, 260), conf_thresh=conf_thresh)
        cv2.imshow('image', img_raw[:,:,::-1])
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--proto', type=str, default='models/face_mask_detection.prototxt', help='prototxt path')
    parser.add_argument('--model', type=str, default='models/face_mask_detection.caffemodel', help='model path')
    args = parser.parse_args()

    Net = cv2.dnn.readNet(args.model, args.proto)
    run_on_video(Net, 0, conf_thresh=0.5)   # video_path should be 0 or 1
