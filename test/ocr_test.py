# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import random
import string
from StringIO import StringIO
from time import time

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageEnhance
from captcha.image import ImageCaptcha

# alphabet = string.digits + string.letters
alphabet = string.letters
file_path = os.path.dirname(os.path.realpath(__file__))
width, height = 160, 60
wx_url = "https://mp.weixin.qq.com/mp/verifycode?cert=1.50902306821e+12&nettype=WIFI&version=12020610&ascene=0&fontScale=100&pass_ticket=5SxzShVEZOSC%2BTzbIKI31zGcCnY27DTO7k4ZnIcO5DlhmnF1YJfVcZ%2BNWNu12%2FCg"
sogo_url = "http://weixin.sogou.com/antispider/util/seccode.php?tc=1509022582371"
data_dir = os.path.join(file_path, "data")


def fetch_img(url=wx_url):
    try:
        content = requests.get(url).content
        img = Image.open(StringIO(content))
        return img
    except Exception as e:
        print(e.message)


def train_set_gen(img_w=width, img_h=height, source_url=wx_url):
    """
    微信、搜狗验证码生成器
    :param img_h:
    :param img_w:
    :param source_url:
    :return:
    """
    while 1:
        img = fetch_img(source_url)
        X = np.zeros((1, img_w, img_h, 1), dtype=np.float32)
        if isinstance(img, Image.Image):
            # img = reduce_noise(img, False)
            # print(img.size)
            if img.size != (img_w, img_h):
                img = img.resize((img_w, img_h), Image.ANTIALIAS)
            tmp_X = np.asarray(img, dtype=np.float32) / 255
            tmp_X = tmp_X.swapaxes(0, 1)
            # 三通道转单通道
            tmp_X = tmp_X[:, :, 0]
            tmp_X = np.expand_dims(tmp_X, 2)
            X[0] = tmp_X
            yield X, img
        else:
            yield None


def save_predict(img, name, type):
    if type == "wx":
        path = os.path.join(data_dir, "wx")
    else:
        path = os.path.join(data_dir, "sogo")
    img_path = os.path.join(path, "%d-%s.jpeg" % (int(round(time() * 1000)), name))
    img.save(img_path)


def fetch_captcha(url=wx_url, count=1000):
    """
    抓取验证码图片
    :return:
    """
    i = 0
    g = train_set_gen(url)
    while i < count:
        next(g)
        i += 1
    print("done! fetch %d captchas" % count)


def walk_captcha():
    img_dir = os.path.join(data_dir, "wx")
    for f in os.listdir(img_dir):
        print(f)
        img_path = os.path.join(img_dir, f)
        img = Image.open(img_path, 'r')
        yield img


def reduce_noise(im, verbose=False):
    """
    图片预处理
    增加对比度 -> 灰度化 -> 二值化 -> 去除离散点
    :return:
    """
    if verbose:
        im.show()
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert("L")
    # im = im.convert('1')
    for i in range(5):
        im = im.filter(ImageFilter.MedianFilter)
    if verbose:
        im.show()
    return im


def enhance_contrast(im):
    """
    增强对比度
    :param im:
    :return:
    """
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(3)
    return im


def random_img(captcha_len=4, img_w=width, img_h=height):
    """
    利用ImageCaptcha库生成随机验证码图片
    :param img_h:
    :param img_w:
    :param captcha_len: 验证码长度
    :return: 验证码图片和内容
    """
    generator = ImageCaptcha(width=img_w, height=img_h, fonts=[os.path.join(file_path, 'kongxin.ttf')])
    # generator = ImageCaptcha(width=img_w, height=img_h)
    random_str = ''.join([random.choice(alphabet) for j in range(captcha_len)])
    captchas = random_str
    img = generator.generate_image(random_str)
    return img, captchas


def gen(batch_size=32, captcha_len=4, img_w=width, img_h=height):
    """
    :param img_h:
    :param img_w:
    :param batch_size:
    :param captcha_len:
    :return:
    """
    X = np.zeros((batch_size, img_w, img_h, 1), dtype=np.float32)
    y = np.ones((batch_size, 16), dtype=np.float32) * -1
    random_chars = ''
    while True:
        for i in range(batch_size):
            img, random_chars = random_img(captcha_len, img_w, img_h)
            img = enhance_contrast(img)
            # img.show()
            tmp_X = np.asarray(img, dtype=np.float32) / 255
            # 将60 * 160 * 3 转成 160 * 60 * 3
            tmp_X = tmp_X.swapaxes(0, 1)
            # 三通道转单通道
            tmp_X = tmp_X[:, :, 0]
            tmp_X = np.expand_dims(tmp_X, 2)
            X[i] = tmp_X
            for j, ch in enumerate(random_chars):
                y[i, j] = alphabet.find(ch)
        yield X, y, random_chars


def test_pytesser():
    pass


if __name__ == '__main__':
    # g = walk_captcha()
    # pre_process(Image.open("data/sogo/1510032435071.jpeg"))
    # a, _ = random_img()
    # pre_process(a)
    g = gen(batch_size=1, captcha_len=4, img_h=60, img_w=160)
    X, _, _ = next(g)

    # g = train_set_gen(source_url=wx_url)
    # x, img = next(g)
    # img.show()
    # reduce_noise(img, True)
    # img = cv2.imread("./data/sogo/test.png")
    # img = Image.open("./data/sogo/test.jpeg")
    # img = reduce_noise(img, False)
    # img = np.asarray(img, dtype=np.uint8)
    # img = np.expand_dims(img, 2)
    # txt = pytesser.image_to_string(img, lang='eng')
    # print(txt)
    random_img()
