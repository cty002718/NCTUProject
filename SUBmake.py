from Setting.attack_setting import *
import struct
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from Structure import cnn_structure, fc_structure, sub_structure
from FCpredict import compute_accuracy, predict_from_array, predict_from_dir_to_dir
import matplotlib.pyplot as plt


def attack_make():
    image_num = MAKE_ATTACK_NUM
    MNIST = np.zeros((image_num, 784), dtype=float)
    LABEL = np.zeros((image_num, 10), dtype=float)
    ANS = np.zeros((image_num,), dtype=int)
    with open("Data/train-images-idx3-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        magic, num, row, column = struct.unpack_from(">IIII", buf)
        index += struct.calcsize(">IIII")
        for i in range(image_num):
            im = struct.unpack_from(">784B", buf, index)
            index += struct.calcsize(">784B")
            im = np.array(im, dtype='uint8')
            MNIST[i] = im

    MNIST = MNIST / 255.

    with open("Data/train-labels-idx1-ubyte", "rb") as f:
        buf = f.read()
        index = 0
        magic, num = struct.unpack_from(">II", buf)
        index += struct.calcsize(">II")
        for i in range(image_num):
            im = struct.unpack_from(">B", buf, index)
            index += struct.calcsize(">B")
            LABEL[i][im] = 1
            ANS[i] = im[0]

    print(compute_accuracy(MNIST, LABEL))

    '''
    g_sub_make = tf.Graph()
    with g_sub_make.as_default():
        xs = tf.placeholder(tf.float32,[None,784])
        ys = tf.placeholder(tf.float32,[None,10])
        keep_prob = tf.placeholder(tf.float32)
        predict, correct_predict = sub_structure(xs, keep_prob)
        correct_num = tf.equal(correct_predict, tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))
        soft = tf.nn.softmax(predict)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels =  ys)
        gradient = tf.gradients(cross_entropy, xs)

        reduc_ind = list(range(1, len(xs.get_shape())))
        #sign_gradient = gradient / tf.sqrt(tf.reduce_sum(tf.square(gradient),
         #                                          reduction_indices=reduc_ind,
          #                                         keep_dims=True))

        sign_gradient = tf.sign(gradient)
        saver = tf.train.Saver()


    sess = tf.Session(graph=g_sub_make)
    saver.restore(sess, "ATTACK_net/save_net.ckpt")

    feed_dict = {xs: MNIST, ys: LABEL, keep_prob: 1}
    deltax = sess.run(sign_gradient, feed_dict=feed_dict)[0]

    #for i, d in enumerate(deltax):
    #    mean = np.mean(d)
    #    std = np.std(d)
    #    deltax[i] = (d - mean) / std

    #for i, a in enumerate(deltax):
     #   print(a)

    ADV = np.clip(MNIST + 0.1 * deltax, 0, 1)
    ADV2 = np.clip(MNIST + 0.15 * deltax, 0, 1)
    ADV3 = np.clip(MNIST + 0.2 * deltax, 0, 1)
    print(compute_accuracy(ADV, LABEL))
    print(compute_accuracy(ADV2, LABEL))
    print(compute_accuracy(ADV3, LABEL))


    if not os.path.exists('Adv'):
        os.makedirs('Adv')

    count = 0
    #predict_result = predict_from_array(ADV)


    for i, item in enumerate(ADV2):
        if predict_from_array(item.reshape(1,784))[0] != ANS[i]:
            item = 255 * item
            im = item.reshape(28, 28)
            im = im.astype("uint8")
            im = Image.fromarray(im)
            im.save('Adv/adv_%s.bmp' % count, 'bmp')

                #im2 = MNIST[i].reshape(28, 28)
            count += 1

    f, a = plt.subplots(4, 10, figsize=(12, 4))

    for i in range(10):
        a[0][i].imshow(np.reshape(MNIST[i], (28, 28)), plt.cm.gray)
        a[1][i].imshow(np.reshape(ADV[i], (28, 28)), plt.cm.gray)
        a[2][i].imshow(np.reshape(ADV2[i], (28, 28)), plt.cm.gray)
        a[3][i].imshow(np.reshape(ADV3[i], (28, 28)), plt.cm.gray)
        a[0][i].set_xticks([])
        a[0][i].set_yticks([])
        a[1][i].set_xticks([])
        a[1][i].set_yticks([])
        a[2][i].set_xticks([])
        a[2][i].set_yticks([])
        a[3][i].set_xticks([])
        a[3][i].set_yticks([])

    plt.show()

    predict_from_dir_to_dir('Adv')
'''
if __name__ == '__main__':
    attack_make()

