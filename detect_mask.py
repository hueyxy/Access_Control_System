import cv2
import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from PIL import Image, ImageDraw, ImageFont

# 锚配置
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# 推断，批量大小为1，模型输出形状为[1，N，4]，
# 因此，我们将锚的尺寸扩大到[1，anchorrnum，4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
id2chiclass = {0: '您戴了口罩', 1: '您没有戴口罩'}
colors = ((0, 255, 0), (255, 0, 0))

proto = 'masked_detector/face_mask_detection.prototxt'
model = 'masked_detector/face_mask_detection.caffemodel'
Net = cv2.dnn.readNet(model, proto)


def getOutputsNames(net):
    # 获取网络中所有层的名称
    layersNames = net.getLayerNames()
    # 获取输出层的名称，即具有未连接输出的层
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def inference(image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), ):
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=target_shape)
    Net.setInput(blob)
    y_bboxes_output, y_cls_output = Net.forward(getOutputsNames(Net))
    # 删除batch维度，因为batch总是1用于推理。
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # 要加快速度，请执行单类NMS，而不是多类NMS。
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep-idx是nms之后的活动边界框。
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    dets = []
    class_id = 1
    for idx in keep_idxs:
        # conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        det = list((xmin, ymin, xmax, ymax))
        dets.append(det)

    return dets, class_id

    # cv2.imshow('image', img_raw[:, :, ::-1])
