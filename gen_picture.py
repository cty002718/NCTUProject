import struct
import numpy as np
from PIL import Image
import os
from Setting.gen_picture_setting import *


def gen(num):
    if not os.path.exists(GEN_PICTURE_DIRNAME):
        os.makedirs(GEN_PICTURE_DIRNAME)

    with open("Data/train-images-idx3-ubyte", "rb") as f:
        count = 0
        buf = f.read()
        index = 0
        magic, maxnum, row, column = struct.unpack_from(">IIII", buf)
        index += struct.calcsize(">IIII")
        if num > maxnum: 
            print('Generating too many pictures!')
        for image in range(num):
            im = struct.unpack_from(">784B", buf, index)
            index += struct.calcsize(">784B")
            im = np.array(im, dtype='uint8')
            im = im.reshape(28,28)
            im = Image.fromarray(im)
            im.save(GEN_PICTURE_DIRNAME + '/mnist_%s.bmp'%count,'bmp')
            count += 1
    print('Generating dataset picture is successful.')
  

if __name__ == '__main__':
    gen(1000)
