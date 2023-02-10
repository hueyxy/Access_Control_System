# FMA-3D
一种在非遮罩人脸图像上添加遮罩的方法。给定真实掩蔽面部图像（a）和非掩蔽面部（d），我们使用来自（a）的掩模和来自（d）的面部区域合成照片逼真的掩模面部图像。
![image](Data/images/FMA-3D.jpg)

# FMA-3D的一些结果
![image](Data/images/mask-sample.jpg)

# Requirements
* python >= 3.7.1
* pytorch >= 1.1.0

# Usage
* Extract the landmarks.
You can extract the 106 landmarks by our [face sdk](../face_sdk) or any other methods.
*在人脸图像上添加遮罩。
You can refer to [add_mask_one.py](add_mask_one.py) as an example.
```sh
python add_mask_one.py
```

# Speed Up
Some advice:
* 通过多重处理编写整个过程。
* Write the function of render in [face_masker.py](face_masker.py) by c++.

# Reference  
This project is mainly inspired by [PRNet](https://github.com/YadiraF/PRNet).
