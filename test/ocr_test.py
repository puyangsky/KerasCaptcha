# -*- coding: utf-8 -*-

import random
from captcha.image import ImageCaptcha
import numpy as np
import os
import string

alphabet = string.digits + string.letters
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


def gen(batch_size=32, captcha_len=4):
    X = np.zeros((batch_size, width, height, 1), dtype=np.float32)
    y = np.ones((batch_size, 16), dtype=np.float32) * -1
    generator = ImageCaptcha(width=width, height=height)
    captchas = ''
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(alphabet) for j in range(captcha_len)])
            captchas = random_str
            img = generator.generate_image(random_str)
            tmp_X = np.asarray(img, dtype=np.float32) / 255
            # 将60 * 160 * 3 转成 160 * 60 * 3
            tmp_X = tmp_X.swapaxes(0, 1)
            # 三通道转单通道
            tmp_X = tmp_X[:, :, 0]
            tmp_X = np.expand_dims(tmp_X, 2)
            X[i] = tmp_X
            for j, ch in enumerate(random_str):
                y[i, j] = alphabet.find(ch)
        yield X, y, captchas


if __name__ == '__main__':
    g = gen(batch_size=1)
    X, y, c = next(g)
    print(X.shape)
    print(X[0, :, :, 0].shape)
    print(y)
    print(c)
