import random

import cv2
import numpy as np
import pymysql
import json
import numpy  # 数据处理的库 Numpy
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import requests

from Silent_Anti_Spoofing import predict_anti
from utils.face_recognition import return_face_recognition_result_mask, return_face_recognition_result_nonmask

import detect_mask
import dlib

mysql_host = "118.178.89.223"
mysql_post = 3306
mysql_user = "hugh"
mysql_password = "123jack"
mysql_db = "attendance_system"
# login_url = "http://118.178.89.223:11453/login/"
login_url = "http://127.0.0.1:8080/login/"
login_header = {'Content-Type': 'application/json'}
login_username = "hueyy"
login_password = "huey123456"
# upload_url = "http://118.178.89.223:11453/access_control/"
upload_url = "http://127.0.0.1:8080/access_control/"
# 判断识别阈值，欧式距离小于0.4即可判断为相似，越小越相似
threshold = 0.4
colors = ((0, 255, 0), (255, 0, 0))
id2chiclass = {0: '您戴了口罩', 1: '您没有戴口罩'}
temperature_u = 37.3
temperature_l = 36.2
# OpenCv 调用摄像头 use camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(r"D:\Users\xiuyu\Desktop\w.mp4")
# 设置视频参数 set camera
cap.set(3, 480)


def return_euclidean_distance(feature_1, feature_2):
    # """
    # 计算两个512D特征向量间的欧式距离并返回
    # :param feature_1: 512D特征向量1
    # :param feature_2: 512D特征向量2
    # :return: 欧式距离
    # """
    # feature_1 = numpy.array(feature_1)
    # feature_2 = numpy.array(feature_2)
    # # 欧式距离计算方法：
    # #   先计算两个特征的差，再对每个子数进行平方处理，将平方处理后子数相加，最后对相加值开平方根
    # dist = numpy.sqrt(numpy.sum(numpy.square(feature_1 - feature_2)))
    # # dist = numpy.sqrt(sum((numpy.array(feature_1)-numpy.array(feature_2))**2))
    source = 0
    try:
        if feature_1.shape and feature_2.shape:
            source = np.dot(feature_1, feature_2)
    except:
        pass
    dist = source
    return dist


def get_user_face_data(get_time, data):
    """
    从数据库获取人脸数据(缓存五分钟)
    :param get_time: 上一次获取数据的时间
    :param data: 上次获取数据的内容
    :return: 获取数据的时间,获取数据的内容
    """
    global temperature_u
    global temperature_l
    if get_time == "" or get_time < (datetime.now() - timedelta(minutes=5)):
        # 连接数据库
        conn = pymysql.connect(
            host=mysql_host,
            port=mysql_post,
            user=mysql_user,
            password=mysql_password,
            db=mysql_db,
            charset='utf8'
        )
        # 创建游标
        cur = conn.cursor()
        sql = 'SELECT user_id,username,first_name,last_name,features,features_mask FROM users_userface INNER JOIN users_user ON users_userface.user_id = users_user.id;'

        # 执行语句
        cur.execute(sql)
        cur2 = conn.cursor()
        sql2 = 'SELECT content FROM system_setting_systemsetting where id >12'
        cur2.execute(sql2)
        # 关闭连接池
        cur.close()
        cur2.close()
        conn.close()
        cur_data = ()

        T = cur2.fetchall()
        temperature_u = float(T[0][0])
        temperature_l = float(T[1][0])

        for idx, user in enumerate(cur.fetchall()):
            feature = np.array(eval(user[4]), dtype=np.float32)
            feature_mask = np.array(eval(user[5]), dtype=np.float32)
            cur_data = cur_data + ((user[:4] + (feature, feature_mask,)),)
        return datetime.now(), cur_data
        # return datetime.now(), data
    return get_time, data


def change_cv2_draw(image, strs, mask, local, local2, sizes, colour):
    # 用于解决 OpenCV 绘图时中文出现乱码 方法类
    '''
        思路：
            1、OpenCV图片格式转换成PIL的图片格式；
            2、使用PIL绘制文字；
            3、PIL图片格式转换成OpenCV的图片格式；
        参数：
            image：OpenCV图片格式的图片
            strs：需要绘制的文字的内容
            local：需要绘制的文字的位置
            sizes：需要绘制的文字的大小
            colur：需要绘制的文字的颜色
    '''
    # 把图片改为rgb顺序图
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 把OpenCV图片格式转换成PIL的图片格式
    pilimg = Image.fromarray(cv2img)
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(pilimg)
    # 字体的样式
    font = ImageFont.truetype("data/font/msyh.ttc", sizes, encoding="utf-8")

    # 绘制文字
    draw.text(local2, id2chiclass[mask], (0, 0, 0), font=font)
    draw.text(local, strs, colour, font=font)
    # 把PIL图片格式转换成OpenCV的图片格式
    image = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)
    return image


# 账户登录
login_post_result = requests.post(url=login_url, headers=login_header,
                                  data=json.dumps({
                                      'username': login_username,
                                      'password': login_password
                                  }))
if json.loads(login_post_result.text)['is_superuser'] is False:
    print('请登录管理员账户！')
    quit()
login_token = json.loads(login_post_result.text)['token']


# 获取温度
def get_temperature():
    temperature = round(random.uniform(36.0, 37.8), 1)
    # if 36.3 < temperature < 37.3:
    #     # print(temperature)
    #     return temperature
    # else:
    #     return False
    return temperature


# 保存图片
def save_image(image):
    save_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '.jpg'
    cv2.imwrite('temp/' + save_name, image)
    return save_name


# 上传图片
def upload_image(save_name, user_id, euclidean_distance, temperature):
    with open('temp/' + save_name, mode="rb", ) as f:  # 打开文件
        file = {
            "file": (save_name, f.read())  # 引号的file是接口的字段，后面的是文件的名称、文件的内容
        }
    upload_post_result = requests.post(
        url=upload_url + '?user_id=' + str(user_id) + '&euclidean_distance=' + str(
            euclidean_distance) + '&temperature=' + str(temperature),
        headers={
            'Authorization': 'JWT ' + login_token
        },
        files=file)

    result = json.dumps(upload_post_result.text)
    print("[BACKEND]", json.loads(result, strict=False))


# 初始化数组等
get_time = datetime.strptime("2020-01-01", "%Y-%m-%d")
calculate_time = datetime.strptime("2020-01-01", "%Y-%m-%d")
data = ""
face_descriptor_old = ""
two_sec_min_euclidean_distance_list = []
two_sec_min_user_info_list = []
two_sec_min_img_list = []
name = ""
color = (0, 0, 0)

# 摄像头循环
while cap.isOpened():
    flag, img_read = cap.read()
    kk = cv2.waitKey(1)
    # img_gray = cv2.cvtColor(img_read, cv2.COLOR_RGB2GRAY)
    img_gray = img_read
    # 人脸数 faces
    # faces = detector(img_gray, 0)
    pred, mask_or_not = detect_mask.inference(img_gray, conf_thresh=0.5, target_shape=(260, 260))  # [x y w h conf]
    faces = []
    # if pred.shape[0]:
    if pred:
        for det in pred:
            # pos = list(map(int, det[:4]))  # 检测到目标位置，格式：（left，top，w，h）
            faces.append(dlib.rectangle(det[0], det[1], det[2], det[3]))
            # faces.append(list((det[0], det[1], det[2], det[3])))
    # 待会要写的字体 / font to write
    font = cv2.FONT_HERSHEY_COMPLEX

    if len(faces) != 0:
        # 矩形框
        # show the rectangle box
        for k, d in enumerate(faces):
            # 判断是否超出边界
            if d.right() > 640 or d.bottom() > 480 or d.left() < 0 or d.top() < 0:
                # if d[2] > 640 or d[3] > 480 or d[0] < 0 or d[1] < 0:
                name = "请保证人脸不要超出边界"
                color = (255, 0, 0)
            # 判断是否多个人
            elif len(faces) != 1:
                name = "请保证摄像头范围内只存在一人"
                color = (255, 0, 0)
            # elif predict_anti(img_gray, pred[0], 0) != 1:  # 1为活体
            #     name = "请不要作弊"
            #     color = (255, 0, 0)
            else:
                # # 生成单张人脸图像的512D特征
                # face_descriptor = face_rec.compute_face_descriptor(img_read, shape)
                if mask_or_not == 1:
                    face_descriptor = return_face_recognition_result_nonmask(img_gray, dets=pred)
                else:
                    face_descriptor = return_face_recognition_result_mask(img_gray, dets=pred)
                # 从数据库获取人脸数据(缓存五分钟)
                get_time, data = get_user_face_data(get_time, data)

                euclidean_distance_list = []
                if mask_or_not == 1:
                    for one_data in data:
                        euclidean_distance_list.append(
                            # return_euclidean_distance(np.array(eval(one_data[4]), dtype=np.float32), face_descriptor))
                            return_euclidean_distance(one_data[4], face_descriptor))
                else:
                    for one_data in data:
                        euclidean_distance_list.append(
                            return_euclidean_distance(one_data[5], face_descriptor))

                """
                if face_descriptor_old != "":
                    if return_euclidean_distance(face_descriptor_old, face_descriptor) > 0.4:
                        print("检测到摄像头前换人了。")
                        name = ""
                        two_sec_min_euclidean_distance_list = []
                        two_sec_min_user_info_list = []
                        two_sec_min_img_list = []
                face_descriptor_old = face_descriptor
                """
                # print(mask_detect(img_gray))
                if calculate_time < (datetime.now() - timedelta(seconds=30)):
                    name=''
                    if two_sec_min_euclidean_distance_list:
                        Temperature = get_temperature()
                        print(max(two_sec_min_euclidean_distance_list), Temperature)
                        if max(two_sec_min_euclidean_distance_list) > threshold:
                            index = two_sec_min_euclidean_distance_list.index(max(two_sec_min_euclidean_distance_list))
                            print("[MESSAGE]识别成功，你可能是：", two_sec_min_user_info_list[index][2],
                                  two_sec_min_user_info_list[index][3], "，余弦分数：",
                                  max(two_sec_min_euclidean_distance_list), "。", id2chiclass[mask_or_not])
                            save_name = save_image(two_sec_min_img_list[index])
                            upload_image(save_name, two_sec_min_user_info_list[index][0],
                                         max(two_sec_min_euclidean_distance_list), Temperature)
                            if temperature_u > Temperature > temperature_l:
                                name = two_sec_min_user_info_list[index][2] + two_sec_min_user_info_list[index][
                                    3] + "-体温正常 "
                                color = (0, 255, 0)
                            else:
                                name = "体温检测异常!"
                                color = (255, 0, 0)
                                # print("Open!")
                        else:

                            # print("[MESSAGE]识别失败，请完善自己账户人脸信息或联系管理员！")
                            name = "识别失败！"
                            # print("Close")
                            color = (255, 0, 0)
                    two_sec_min_euclidean_distance_list = []
                    two_sec_min_user_info_list = []
                    two_sec_min_img_list = []
                    calculate_time = datetime.now()
                else:
                    two_sec_min_euclidean_distance_list.append(max(euclidean_distance_list))
                    two_sec_min_user_info_list.append(data[euclidean_distance_list.index(max(euclidean_distance_list))])
                    two_sec_min_img_list.append(img_read)

            # 画边框
            cv2.rectangle(img_read,
                          tuple([d.left(), d.top()]),
                          tuple([d.right(), d.bottom()]),
                          # tuple([d[0], d[1]]),
                          # tuple([d[2], d[3]]),
                          colors[mask_or_not], 2)
            # 文字位置
            pos_namelist_mask = tuple([d.left() + 5, d.bottom() - 25])
            pos_namelist = tuple([d.left() - 5, d.top() - 25])
            # pos_namelist = tuple([d[0] + 5, d[3] - 25])
            img_read = change_cv2_draw(img_read, name, mask_or_not, pos_namelist, pos_namelist_mask, 16, color)
    cv2.imshow("Dormitory Access Control System(Camera)", img_read)
    kk = cv2.waitKey(1)
