"""
from users.models import User
user = User.objects.filter()
print(user)


from users.models import User
User.objects.get(username='hueyy', is_superuser=True).delete()

from users.models import User

user = User.objects.create_user('hueys', '2079111111@qq.com', 'huey123456')
"""
"""
# http:
# users.models.User.info.RelatedObjectDoesNotExist: User has no info.
# require
# GET '/users/?page=1&limit=10&is_active=false&is_staff=true&search_firstname=&search_lastname=&search_username
# =&search_mobile=&search_email='>
#
# GET '/users/?page=1&limit=10&is_active=true&is_staff=true&search_firstname=&search_lastname=&search_username
# =&search_mobile=&search_email='>
#
#
#
# # """
from datetime import datetime

import numpy as np
import pymysql

conn = pymysql.connect(
    host="118.178.89.223",
    port=3306,
    user="hugh",
    password="123jack",
    db="attendance_system",
    charset='utf8'
)

conn2 = pymysql.connect(
    host="118.178.89.223",
    port=3306,
    user="hugh",
    password="123jack",
    db="aiface",
    charset='utf8'
)
# 创建游标
cur = conn.cursor()
# sql = 'SELECT code,content FROM system_setting_systemsetting where id >12'
# cur = conn.cursor()
cur2 = conn2.cursor()
# sql = 'SELECT user_id,username,first_name,last_name,features FROM users_userface INNER JOIN users_user ON users_userface.user_id = users_user.id;'
# cur.execute(sql)
# print(cur.fetchall())
sql2 = "select * from features left join user on features.uid = user.uid"
cur2.execute(sql2)
features_all = cur2.fetchall()
features = []
features_mask = []
for i in features_all:
    if i[3] == 1:
        fea = np.array(eval(i[4]), dtype=np.float32)
        features_mask.append(fea)
    else:
        fea = np.array(eval(i[4]), dtype=np.float32)
        features.append(fea)

# cur.execute(sql)
# 关闭连接池
cur2.close()
conn2.close()
for fea1, fea2 in zip(features, features_mask):
    sql = "insert into users_userface(add_time,features,features_mask) values ('%s','%s','%s')" % (
        datetime.now(), fea1.tolist(), fea2.tolist())
    cur.execute(sql)
    conn.commit()

# 执行语句
# cur.execute(sql)
# 关闭连接池
cur.close()
conn.close()

# data = ()
# for idx, user in enumerate(cur.fetchall()):
#     feature = np.array(eval(user[4]), dtype=np.float32)
#     data = data + ((user[:4] + (feature,)),)
#
# for d in data:
#     print(d[4])
#     print(type(d[4]))
#     # print(user[:4] + (np.array(eval(user[4]), dtype=np.float32),))
#     # data = data + (user[:4] + (np.array(eval(user[4]), dtype=np.float32),),)
# print(data)
# print(type(data))
# for date in cur.fetchall():
# name.append(i)
# fea = np.array(eval(i[4]), dtype=np.float32)
# print(type(fea))
# features.append(fea)
# import cv2
# import detect_mask
#
# img = cv2.imread(r"E:\Users\ASUS\Pictures\Saved Pictures\123.jpg")
# print(detect_mask.inference(img, conf_thresh=0.5, target_shape=(260, 260)))

#####################################################################################################################
#
# import cv2
# import numpy as np
# import pymysql
# import json
# import dlib
# import numpy  # 数据处理的库 Numpy
# from datetime import datetime, timedelta
# from PIL import Image, ImageDraw, ImageFont
# import requests
# # from utils.face_recognition import return_face_recognition_result_mask, return_face_128d_features, \
# #     return_face_recognition_result_nonmask
# # from GetFeatures import faceDetModelHandler_mask
# import detect_mask
# from Silent_Anti_Spoofing import predict_anti
#
# mysql_host = "118.178.89.223"
# mysql_post = 3306
# mysql_user = "hugh"
# mysql_password = "123jack"
# mysql_db = "attendance_system"
# # login_url = "http://118.178.89.223:11453/login/"
# login_url = "http://127.0.0.1:8080/login/"
# login_header = {'Content-Type': 'application/json'}
# login_username = "hueyy"
# login_password = "huey123456"
# # upload_url = "http://118.178.89.223:11453/access_control/"
# upload_url = "http://127.0.0.1:8080/access_control/"
# # 判断识别阈值，欧式距离小于0.4即可判断为相似，越小越相似
# threshold = 0.4
#
# # OpenCv 调用摄像头 use camera
# cap = cv2.VideoCapture(r"D:\Users\xiuyu\Downloads\Video\5.mp4")
# print(cap.isOpened())
# # 设置视频参数 set camera
# cap.set(3, 480)
#
#
# # def return_euclidean_distance(feature_1, feature_2):
# #     # """
# #     # 计算两个128D特征向量间的欧式距离并返回
# #     # :param feature_1: 128D特征向量1
# #     # :param feature_2: 128D特征向量2
# #     # :return: 欧式距离
# #     # """
# #     # feature_1 = numpy.array(feature_1)
# #     # feature_2 = numpy.array(feature_2)
# #     # # 欧式距离计算方法：
# #     # #   先计算两个特征的差，再对每个子数进行平方处理，将平方处理后子数相加，最后对相加值开平方根
# #     # dist = numpy.sqrt(numpy.sum(numpy.square(feature_1 - feature_2)))
# #     # # dist = numpy.sqrt(sum((numpy.array(feature_1)-numpy.array(feature_2))**2))
# #     source = 0
# #     try:
# #         if feature_1.shape and feature_2.shape:
# #             source = np.dot(feature_1, feature_2)
# #     except:
# #         pass
# #     dist = source
# #     return dist
# #
# #
# def change_cv2_draw(image, strs, mask, local, sizes, colour):
#     # 用于解决 OpenCV 绘图时中文出现乱码 方法类
#     '''
#         思路：
#             1、OpenCV图片格式转换成PIL的图片格式；
#             2、使用PIL绘制文字；
#             3、PIL图片格式转换成OpenCV的图片格式；
#         参数：
#             image：OpenCV图片格式的图片
#             strs：需要绘制的文字的内容
#             local：需要绘制的文字的位置
#             sizes：需要绘制的文字的大小
#             colur：需要绘制的文字的颜色
#     '''
#     # 把图片改为rgb顺序图
#     cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # 把OpenCV图片格式转换成PIL的图片格式
#     pilimg = Image.fromarray(cv2img)
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(pilimg)
#     # 字体的样式
#     font = ImageFont.truetype("data/font/msyh.ttc", sizes, encoding="utf-8")
#     id2chiclass = {0: '您戴了口罩', 1: '您没有戴口罩'}
#     # 绘制文字
#     draw.text(local, strs + "---" + id2chiclass[mask], colour, font=font)
#     # 把PIL图片格式转换成OpenCV的图片格式
#     image = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)
#     return image
#
#
# # 初始化数组等
# get_time = datetime.strptime("2020-01-01", "%Y-%m-%d")
# calculate_time = datetime.strptime("2020-01-01", "%Y-%m-%d")
# data = ""
# face_descriptor_old = ""
# two_sec_min_euclidean_distance_list = []
# two_sec_min_user_info_list = []
# two_sec_min_img_list = []
# name = ""
# color = (0, 0, 0)
#
# # 摄像头循环
# while cap.isOpened():
#     flag, img_read = cap.read()
#     kk = cv2.waitKey(1)
#
#     # img_gray = cv2.cvtColor(img_read, cv2.COLOR_RGB2GRAY)
#     img_gray = img_read
#     # 人脸数 faces
#     # faces = detector(img_gray, 0)
#     pred, mask_or_not = detect_mask.inference(img_gray, conf_thresh=0.5, target_shape=(260, 260))  # [x y w h conf]
#     faces = []
#     # if pred.shape[0]:
#     if pred:
#         for det in pred:
#             # pos = list(map(int, det[:4]))  # 检测到目标位置，格式：（left，top，w，h）
#             # faces.append(dlib.rectangle(pos[0], pos[1], pos[2], pos[3]))
#             faces.append(dlib.rectangle(det[0], det[1], det[2], det[3]))
#     print("pred=", pred)
#     # 待会要写的字体 / font to write
#     font = cv2.FONT_HERSHEY_COMPLEX
#
#     if len(faces) != 0:
#         # 矩形框
#         # show the rectangle box
#         for k, d in enumerate(faces):
#             # 判断是否超出边界
#             if d.right() > 640 or d.bottom() > 480 or d.left() < 0 or d.top() < 0:
#                 name = "请保证人脸不要超出边界。"
#                 color = (255, 0, 0)
#             # 判断是否多个人
#             elif len(faces) != 1:
#                 name = "请保证摄像头范围内只存在一人。"
#                 color = (255, 0, 0)
#             else:
#                 # # 采用 Dlib 的人脸5特征点检测器
#                 # shape = predictor(img_gray, d)
#                 # # 生成单张人脸图像的512D特征
#                 # face_descriptor = face_rec.compute_face_descriptor(img_read, shape)
#                 # if mask_or_not == "Nomask":
#                 #     face_descriptor = return_face_recognition_result_nonmask(img_gray, dets=pred)
#                 # else:
#                 #     face_descriptor = return_face_recognition_result_mask(img_gray, dets=pred)
#                 # 从数据库获取人脸数据(缓存五分钟)
#
#                 euclidean_distance_list = []
#                 # if mask_or_not == 1:
#                 #     face_descriptor = return_face_recognition_result_nonmask(img_gray, dets=pred)
#                 # else:
#                 #     face_descriptor = return_face_recognition_result_mask(img_gray, dets=pred)
#                 #
#                 # if mask_or_not == 1:
#                 #     for one_data in features:
#                 #         euclidean_distance_list.append(
#                 #             # return_euclidean_distance(np.array(eval(one_data[4]), dtype=np.float32), face_descriptor))
#                 #             return_euclidean_distance(one_data, face_descriptor))
#                 # else:
#                 #     for one_data in features_mask:
#                 #         euclidean_distance_list.append(
#                 #             return_euclidean_distance(one_data, face_descriptor))
#
#                 """
#                 if face_descriptor_old != "":
#                     if return_euclidean_distance(face_descriptor_old, face_descriptor) > 0.4:
#                         print("检测到摄像头前换人了。")
#                         name = ""
#                         two_sec_min_euclidean_distance_list = []
#                         two_sec_min_user_info_list = []
#                         two_sec_min_img_list = []
#                 face_descriptor_old = face_descriptor
#                 """
#                 # print(mask_detect(img_gray))
#                 # if calculate_time < (datetime.now() - timedelta(seconds=2)):
#                 #     if two_sec_min_euclidean_distance_list:
#                 #         if max(two_sec_min_euclidean_distance_list) > threshold:
#                 #             index = two_sec_min_euclidean_distance_list.index(max(two_sec_min_euclidean_distance_list))
#                 #             print("[MESSAGE]识别成功，你可能是：", "，欧式距离：",
#                 #                   max(two_sec_min_euclidean_distance_list), "。", mask_or_not)
#                 #
#                 #             name = "Hello"
#                 #             color = (0, 255, 0)
#                 #         else:
#                 #             print("[MESSAGE]识别失败，请完善自己账户人脸信息或联系管理员！")
#                 #             name = "识别失败，请完善自己账户人脸信息或联系管理员！"
#                 #             color = (255, 0, 0)
#                 #     two_sec_min_euclidean_distance_list = []
#                 #     two_sec_min_user_info_list = []
#                 #     two_sec_min_img_list = []
#                 #     calculate_time = datetime.now()
#                 # else:
#                 #     two_sec_min_euclidean_distance_list.append(max(euclidean_distance_list))
#                 #
#                 #     two_sec_min_img_list.append(img_read)
#
#             # 画边框
#             cv2.rectangle(img_read,
#                           tuple([d.left(), d.top()]),
#                           tuple([d.right(), d.bottom()]),
#                           (200, 0, 0), 2)
#             # 文字位置
#             pos_namelist = tuple([d.left() + 5, d.bottom() - 25])
#             img_read = change_cv2_draw(img_read, name, mask_or_not, pos_namelist, 16, color)
#     cv2.imshow("Dormitory Access Control System(Camera)", img_read)
#     kk = cv2.waitKey(1)
