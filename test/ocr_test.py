# -*- coding: utf-8 -*-

import random
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import os

alphabet = 'abcdefghijklmnopqrstuvwxyz'
data_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
width, height = 160, 60


def image_gen(batch_size=10, img_w=160, img_h=60):
    generator = ImageCaptcha(width=width, height=height)
    i = 0
    f = open(data_path + "text.txt", 'a+')
    while i < batch_size:
        text = "".join(random.sample(alphabet, 4))
        f.write("%s\n" % text)
        img_name = str(i) + '-' + text + ".png"
        path = data_path + img_name
        # img = generator.generate(text)
        generator.write(text, path)
        i += 1
    f.close()
    print "generate %d captchas" % batch_size


def test_gen(batch_size=10, img_w=160, img_h=60):
    # image_gen(batch_size=batch_size, img_w=img_w, img_h=img_h)
    # f = open(data_path + "text.txt", 'r')
    X = np.zeros((batch_size, img_h, img_w, 1), dtype=np.float32)
    y = []
    # for i, line in enumerate(f.readlines()):
    #     text = line.strip('\n')
    #     img_name = "%d-%s.png" % (i, text)
    #     img = Image.open(data_path + img_name)
    #     img = img.convert("I")
    #     test_x = np.asarray(img, dtype=np.float32) / 255
    #     y = text
    #     yield (test_x, y)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            text = "".join(random.sample(alphabet, 4))
            img = generator.generate_image(text)
            tmp_X = np.asarray(img, dtype=np.float32) / 255
            for index_i, tmp_i in enumerate(tmp_X):
                for index_j, tmp_j in enumerate(tmp_i):
                    if index_j >= img_w or index_i >= img_h:
                        continue
                    # 三通道转单通道
                    X[i][index_i][index_j] = tmp_j[0]
            Image._show(img)
            y.append(text)
        yield X, y


if __name__ == '__main__':
    gen = test_gen(batch_size=1, img_h=10, img_w=10)
    x, y = next(gen)
    f = open("test.txt", 'w')
    for i in x[0]:
        for j in i:
            f.write("%.2f\t" % (j[0] - 0.992157))
        f.write('\n')
    f.close()
    print y