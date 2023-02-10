import os
import sys

import cv2
import numpy as np
import torch
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

# 人脸检测模型设置。
device = 'cpu'
# if torch.cuda.is_available():
#     device

faceDetModelLoader = FaceDetModelLoader('models', 'face_detection', 'face_detection_mask')
model, cfg = faceDetModelLoader.load_model()
faceDetModelHandler_mask = FaceDetModelHandler(model, device, cfg)
faceDetModelLoader = FaceDetModelLoader('models', 'face_detection', 'face_detection_nonmask')
model, cfg = faceDetModelLoader.load_model()
faceDetModelHandler_nonmask = FaceDetModelHandler(model, device, cfg)

# face landmark model setting.

faceAlignModelLoader = FaceAlignModelLoader('models', 'face_alignment', 'face_alignment_nonmask')
model, cfg = faceAlignModelLoader.load_model()
faceAlignModelHandler_nonmask = FaceAlignModelHandler(model, device, cfg)

faceAlignModelLoader = FaceAlignModelLoader('models', 'face_alignment', 'face_alignment_mask')
model, cfg = faceAlignModelLoader.load_model()
faceAlignModelHandler_mask = FaceAlignModelHandler(model, device, cfg)

# face recognition model setting.

faceRecModelLoader = FaceRecModelLoader('models', 'face_recognition', 'face_recognition_mask')
model, cfg = faceRecModelLoader.load_model()
faceRecModelHandler_mask = FaceRecModelHandler(model, device, cfg)
faceRecModelLoader = FaceRecModelLoader('models', 'face_recognition', "face_recognition_nonmask")
model, cfg = faceRecModelLoader.load_model()
faceRecModelHandler_nonmask = FaceRecModelHandler(model, device, cfg)

# 读取图像并获取面部特征。
face_cropper = FaceRecImageCropper()
#
# if __name__ == "__main__":
#     feature_list = []
#     # list_dir = os.listdir("E:/PycharmProjects/project_Qt/photo/")
#     # for file_name in tqdm(list_dir):
#     image_paths = "E:/PycharmProjects/project_Qt/photo/chenduling_1_y_n.jpg"
#     image = cv2.imread(image_paths, cv2.IMREAD_COLOR)
#     dets = faceDetModelHandler_nonmask.inference_on_image(image)
#     landmarks = faceAlignModelHandler_mask.inference_on_image(image, dets[0])
#     landmarks_list = []
#     for (x, y) in landmarks.astype(np.int32):
#         landmarks_list.extend((x, y))
#     cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
#     cv2.imwrite("cro.jpg", cropped_image)
#     feature = faceRecModelHandler_mask.inference_on_image(cropped_image)
#     print(feature)
#     print(type(feature))
    # feature_list.append(feature)
# for i in range(len(feature_list)):
#     score = np.dot(feature_list[0], feature_list[i])
#     logger.info('两张脸的相似度得分: %f' % score)
# CUDA_VISIBLE_DEVICE
