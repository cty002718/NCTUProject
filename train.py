from PIL import Image

from Setting.train_setting import *
import os
import numpy as np
import tensorflow as tf
from open_data import open_data, get_some_data
from Structure import cnn_structure, fc_structure, sub_structure
import struct
if ATTACK_OBJ != 1:
    from FCpredict import predict_from_array
else:
    from CNNpredict import predict_from_array


def cnn_train():

    g_cnn_train = tf.Graph()
    with g_cnn_train.as_default():
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        predict, correct_prediction = cnn_structure(xs, keep_prob)

        correct_num = tf.equal(correct_prediction, tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=ys))
        train_step = tf.train.AdamOptimizer(CNN_LEARNING_RATE).minimize(cross_entropy)
        saver = tf.train.Saver()

    with tf.Session(graph=g_cnn_train) as sess:
        sess.run(tf.global_variables_initializer())
        train_batch = open_data()
        train_data = np.vstack((train_batch[0], train_batch[1]))
        for i in range(2, 6):
            train_data = np.vstack((train_data, train_batch[i]))

        for i in range(560*CNN_TRAIN_NUM):
            batch_xs = train_batch[i % 560][:, 10:]
            batch_ys = train_batch[i % 560][:, :10]
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: CNN_PROB})
            if i % (56*2) == 0:
                print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}))

        if not os.path.exists("CNN_net"):
            os.makedirs("CNN_net")

        saver.save(sess, "CNN_net/save_net.ckpt")


def fc_train():

    g_fc_train = tf.Graph()
    with g_fc_train.as_default():
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        predict, correct_prediction = fc_structure(xs, keep_prob)

        correct_num = tf.equal(correct_prediction, tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=ys))
        train_step = tf.train.AdamOptimizer(FC_LEARNING_RATE).minimize(cross_entropy)
        saver = tf.train.Saver()

    with tf.Session(graph=g_fc_train) as sess:
        sess.run(tf.global_variables_initializer())
        train_batch = open_data()
        #print(train_batch[0][0])
        train_data = np.vstack((train_batch[0], train_batch[1]))
        for i in range(2, 6):
            train_data = np.vstack((train_data, train_batch[i]))

        for i in range(560*FC_TRAIN_NUM):
            batch_xs = train_batch[i % 560][:, 10:]
            batch_ys = train_batch[i % 560][:, :10]
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: FC_PROB})
            if i % (56*2) == 0:
                print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}))

        if not os.path.exists("FC_net"):
            os.makedirs("FC_net")

        saver.save(sess, "FC_net/save_net.ckpt")


def attack_train():

    g_sub_train = tf.Graph()
    with g_sub_train.as_default():
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        predict, correct_prediction = sub_structure(xs, keep_prob)
        soft = tf.nn.softmax(predict)

        correct_num = tf.equal(correct_prediction, tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=ys))
        train_step = tf.train.AdamOptimizer(ATTACK_LEARNING_RATE).minimize(cross_entropy)
        saver = tf.train.Saver()

    with tf.Session(graph=g_sub_train) as sess:
        image_num = ATTACK_TRAIN_INIT
        mnist = np.zeros((image_num, 784), dtype=float)
        with open("Data/train-images-idx3-ubyte", "rb") as f:
            buf = f.read()
            index = 0
            magic, max_num, row, column = struct.unpack_from(">IIII", buf)
            index += struct.calcsize(">IIII")
            for i in range(image_num):
                im = struct.unpack_from(">784B", buf, index)
                index += struct.calcsize(">784B")
                im = np.array(im, dtype='uint8')
                mnist[i] = im

        mnist = mnist / 255.

        test_data, test_label = get_some_data(1000)
        sess.run(tf.global_variables_initializer())
        now = -1
        for epoc in range(ATTACK_TRAIN_EPOC):
            Label = np.zeros((image_num, 10), dtype=float)
            ANS = predict_from_array(mnist)
            for i in range(image_num):
                Label[i][ANS[i]] = 1

            for i in range(30):
                sess.run(train_step, feed_dict={xs: mnist, ys: Label, keep_prob: ATTACK_PROB})

            print(sess.run(accuracy, feed_dict={xs: test_data, ys: test_label, keep_prob: 1}))

            if epoc == ATTACK_TRAIN_EPOC - 1:
                break

            gradient = [sess.run(tf.gradients(predict[:, i], xs), feed_dict={xs: mnist, keep_prob: 1})[0] for i in range(10)]
            deltax = np.array([gradient[ANS[i]][i] for i in range(image_num)])
            add = ATTACK_LAMBDA * np.sign(deltax)

            if epoc % 2 == 0:
                now = -now
            if now == -1:
                add = -add

            print(now)
            New_image = mnist + add
            mnist = np.vstack((New_image, mnist))
            image_num *= 2
            mnist = np.clip(mnist, 0, 1)

        if not os.path.exists("ATTACK_net"):
            os.makedirs("ATTACK_net")

        saver.save(sess, "ATTACK_net/save_net.ckpt")


if __name__ == '__main__':
    fc_train()
