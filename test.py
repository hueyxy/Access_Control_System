import cv2
import numpy as np
import pymysql
import json
import dlib
import numpy  # 数据处理的库 Numpy
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import requests
from utils.face_recognition import return_face_recognition_result_mask, return_face_128d_features, \
    return_face_recognition_result_nonmask
from GetFeatures import faceDetModelHandler_mask
import detect_mask

import numpy as np
import pymysql

conn2 = pymysql.connect(
    host="118.178.89.223",
    port=3306,
    user="hugh",
    password="123jack",
    db="aiface",
    charset='utf8'
)
# 创建游标
cur2 = conn2.cursor()

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



