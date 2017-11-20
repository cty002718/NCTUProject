import pandas as pd
import numpy as np
import struct


def open_data():
    # 用.csv打開訓練數據
    df = pd.read_csv('Data/train.csv')
    df = pd.concat([pd.get_dummies(df['label'], prefix='label'), df], axis=1)
    df = df.drop(['label'], axis=1)
    train_data1 = df.values
    train_data1 = train_data1.astype("float")
    train_data1[:, 10:] = train_data1[:, 10:]/255.

    # 用test_data打開訓練數據
    Label = np.zeros((10000, 10), dtype=float)
    with open("Data/t10k-labels-idx1-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        magic, num = struct.unpack_from(">II", buf)
        index += struct.calcsize(">II")
        for i in range(num):
            l = struct.unpack_from(">B", buf, index)
            index += struct.calcsize(">B")
            Label[i][l] = 1

    Image = np.zeros((10000, 784), dtype=float)
    with open("Data/t10k-images-idx3-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        magic, num, row, column = struct.unpack_from(">IIII", buf)
        index += struct.calcsize(">IIII")
        for i in range(num):
                im = struct.unpack_from(">784B", buf, index)
                index += struct.calcsize(">784B")
                im = np.array(im, dtype='uint8')
                Image[i] = im

    train_data2 = np.hstack((Label, Image/255.))

    # 用train_data打開訓練數據
    Label = np.zeros((60000, 10), dtype=float)
    with open("Data/train-labels-idx1-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        magic, num = struct.unpack_from(">II", buf)
        index += struct.calcsize(">II")
        for i in range(num):
            l = struct.unpack_from(">B", buf, index)
            index += struct.calcsize(">B")
            Label[i][l] = 1

    Image = np.zeros((60000, 784), dtype=float)
    with open("Data/train-images-idx3-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        magic, num, row, column = struct.unpack_from(">IIII", buf)
        index += struct.calcsize(">IIII")
        for i in range(num):
            im = struct.unpack_from(">784B", buf, index)
            index += struct.calcsize(">784B")
            im = np.array(im, dtype='uint8')
            Image[i] = im

    train_data3 = np.hstack((Label, Image/255.))

    train_data = np.vstack((train_data1, train_data2, train_data3))  # 合併數據
    np.random.shuffle(train_data)  # 打亂數據
    train_batch = np.array_split(train_data, 560)  # 分成 560 個 batch，每個 batch 200 筆資料
    return train_batch


def get_some_data(data_num):
    label = np.zeros((data_num, 10), dtype=float)
    ans = np.zeros((data_num,), dtype=int)
    with open("Data/train-labels-idx1-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        index += struct.calcsize(">II")
        for i in range(data_num):
            l = struct.unpack_from(">B", buf, index)
            index += struct.calcsize(">B")
            label[i][l] = 1
            ans[i] = l[0]

    image = np.zeros((data_num, 784), dtype=float)
    with open("Data/train-images-idx3-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        index += struct.calcsize(">IIII")
        for i in range(data_num):
            im = struct.unpack_from(">784B", buf, index)
            index += struct.calcsize(">784B")
            im = np.array(im, dtype='uint8')
            image[i] = im

    return image/255., label, ans

if __name__ == '__main__':
    open_data()
