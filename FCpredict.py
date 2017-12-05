import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
from Structure import fc_structure

g_fc = tf.Graph()
with g_fc.as_default():
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    predict, correct_prediction = fc_structure(xs, keep_prob)
    correct_num = tf.equal(correct_prediction, tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))
    saver = tf.train.Saver()

sess = tf.Session(graph=g_fc)
if os.path.exists('FC_net'):
    saver.restore(sess, "FC_net/save_net.ckpt")


def predict_from_path(path):
    image = Image.open(path)
    test_data = np.array(image).reshape((1, 784,))
    test_data = test_data.astype("float")
    test_data = test_data / 255.
    output = sess.run(correct_prediction, feed_dict={xs: test_data, keep_prob: 1})
    return output[0]


def predict_from_array(a):
    return sess.run(correct_prediction, feed_dict={xs: a, keep_prob: 1})


# 將圖片分類至不同的資料夾
def predict_from_dir_to_dir(dirname):
    if not os.path.exists("ANS"): os.makedirs("ANS")
    for i in range(10):
        if not os.path.exists("ANS/" + str(i)):
            os.makedirs("ANS/" + str(i))

    dirs = os.listdir(dirname)
    for item in dirs:
        if item == '.DS_Store':
            continue
        image = Image.open('%s/%s' % (dirname, item))
        test_data = np.array(image).reshape((1, 784,))
        test_data = test_data.astype("float")
        test_data = test_data / 255.
        output = sess.run(correct_prediction, feed_dict={xs: test_data, keep_prob: 1})
        image.save("ANS/%s/%s" % (output[0], item), "bmp")


# 將圖片結果輸出至.csv
def predict_from_csv_to_csv():
    df = pd.read_csv('Data/test.csv')
    test_data = df.values
    test_data = test_data.astype("float")
    test_data = test_data / 255.

    output = sess.run(correct_prediction, feed_dict={xs: test_data, keep_prob: 1})
    ans = np.zeros((test_data.shape[0], 2), dtype=int)

    for i in range(output.shape[0]):
        ans[i, 0] = i+1
        ans[i, 1] = output[i]

    df_result = pd.DataFrame(ans, columns=['ImageId', 'Label'])
    df_result.to_csv('Data/MNIST_result.csv', index=False)


def compute_accuracy(x, y):
    return sess.run(accuracy, feed_dict={xs: x, ys:y, keep_prob: 1})

if __name__ == '__main__':
    predict_from_csv_to_csv()
