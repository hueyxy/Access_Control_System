import os
import cv2
import numpy
import dlib
from GetFeatures import *
import detect_mask


def return_face_recognition_result_mask(img, dets):  # 图片帧， 坐标
    feature = []
    if dets:
        try:
            landmarks = faceAlignModelHandler_mask.inference_on_image(img, dets[0])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(img, landmarks_list)
            feature = faceRecModelHandler_mask.inference_on_image(cropped_image)
        except:
            pass

    return feature


def return_face_recognition_result_nonmask(img, dets):  # 图片帧， 坐标
    feature = []
    # dets = faceDetModelHandler_nonmask.inference_on_image(img)# [[][]]
    # if dets.shape[0]:
    if dets:
        try:
            landmarks = faceAlignModelHandler_nonmask.inference_on_image(img, dets[0])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(img, landmarks_list)
            feature = faceRecModelHandler_nonmask.inference_on_image(cropped_image)
        except:
            pass

    return feature


# 根据 参数(图片路径) 返回单张人脸图像的128D特征
def return_face_128d_features(image_path):
    '''
        参数：
            image_path(图片路径)，示例：
                "training_data/qiezi/timg-1.jpg"
        返回：
            face_512d_features(人脸128D特征)，示例：
                "-0.0839738 0.0859853   ... 0.0688491   0.0536951"
    '''

    # 采用 opencv 的 imread 方法根据图片路径参数读取图片
    # img_read = cv2.imdecode(numpy.fromfile(image_path, dtype=numpy.uint8), -1)
    img_read = cv2.imread(image_path)
    # 因 opencv 读取图片默认为 bgr 顺序，这里采用 opencv 的 cvtColor 把图片改为rgb顺序图
    # img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    img_gray = img_read
    # 采用 Dlib 的正向人脸检测器 预先检测人脸情况并存入 faces 数组
    # faces = detector(img_gray, 1)
    # faces = faceDetModelHandler_mask.inference_on_image(img_gray)
    faces, class_mask = detect_mask.inference(img_gray, conf_thresh=0.5, target_shape=(260, 260))
    # 判断检测的图片中是否不存在人脸或出现多张人脸，faces的长度即为检测到人脸的个数
    if len(faces) == 0:
        # 检测不到人脸
        os.remove(image_path)
        return 0
    if len(faces) > 1:
        # 检测人脸数大于2
        os.remove(image_path)
        return 1
    if len(faces) == 1:
        # 如果人脸数为 1
        # 生成单张人脸图像的512D特征
        if class_mask == 1:
            face_512d_features = return_face_recognition_result_nonmask(img_gray, faces)
        else:
            face_512d_features = return_face_recognition_result_mask(img_gray, faces)
        return face_512d_features, class_mask
