from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random


# 输入：水印图像路劲，原图路劲，.jpg格式
# 输出：mask路径
def generate_one_mask(water_image_path, gt_image_path):
    # 读取图像
    water_image = Image.open('images/' + water_image_path + '.jpg')
    gt_image = Image.open(gt_image_path)

    # 转成numpy数组格式
    water_image = 255 - np.array(water_image)[:, :, :3]
    gt_image = 255 - np.array(gt_image)[:, :, :3]

    # 设置阈值
    threshold = 5
    diff_image = np.abs(water_image.astype(np.float32) - gt_image.astype(np.float32))
    mean_image = np.max(diff_image, axis=-1)
    mask = np.greater(mean_image, threshold).astype(np.uint8) * 255
    mask[mask < 2] = 0
    mask[mask >= 1] = 255
    mask = np.clip(mask, 0, 255)

    # # 增强
    # kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask[mask > 150] = 255
    # mask[mask != 255] = 0
    #
    # kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    # mask[mask > 150] = 255

    # 保存
    mask = np.array([mask, mask, mask, mask])
    mask = mask.transpose(1, 2, 0)
    mask = Image.fromarray(mask[:, :, :3])
    mask.save('masks/' + water_image_path + '.jpg')


generate_one_mask('bg_image_00000_0001', 'bg_images/bg_image_00000.jpg')


# 生成所有图片的mask
i = 0
for idx in range(0, 1841):
    # 正则化路劲
    a = idx // 1000
    b = idx % 1000 // 100
    c = idx % 100 // 10
    d = idx % 10 // 1

    for j in range(1, 552):
        e = j // 100
        f = j % 100 // 10
        g = j % 10 // 1
        water_image_path = f'bg_image_0{a}{b}{c}{d}_0{e}{f}{g}'
        gt_image_path = f'bg_images/bg_image_0{a}{b}{c}{d}.jpg'
        """
        生成mask时，我已经将数据集从551缩减至100。因此生成的mask也有184100张，后面我觉得数据集还是太大，就将100再缩减至20。
        """
        try:
            generate_one_mask(water_image_path, gt_image_path)
            i += 1
            print(f'已完成{100 * i / 184100}%')
        except:
            pass