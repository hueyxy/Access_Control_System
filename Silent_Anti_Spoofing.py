import os

import cv2
import numpy as np
from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from Silent_Face_Anti_Spoofing.src.utility import parse_model_name


def predict_anti(image_name, img_bbox, device_id, model_dir="Silent_Face_Anti_Spoofing/resources/anti_spoof_models"):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    # height, width, channel = image_name.shape
    # if width / height != 3 / 4:
    #     image = cv2.resize(image_name, (480, 640))
    # else:
    image = image_name
    image_bbox = [img_bbox[0], img_bbox[1], abs(img_bbox[2] - img_bbox[1]), abs(img_bbox[3] - img_bbox[0])]
    prediction = np.zeros((1, 3))
    # 对单个模型结果的预测求和
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
    # 绘制预测结果
    label = np.argmax(prediction)
    # value = prediction[0][label] / 2
    return label
