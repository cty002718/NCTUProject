from Setting.attack_setting import *
import struct
import os
from open_data import get_some_data
import numpy as np
import tensorflow as tf
from PIL import Image
from Structure import cnn_structure, fc_structure, sub_structure
from FCpredict import compute_accuracy, predict_from_array, predict_from_dir_to_dir
import matplotlib.pyplot as plt


def jsmaOnePicture(sess, image, xs, keep_prob, correct_predict, target, theta, grads):
    adv_x = np.copy(image)

    if theta > 0:
        search_domain = set([i for i in range(784)
                             if adv_x[0, i] < 255])
    else:
        search_domain = set([i for i in range(784)
                             if adv_x[0, i] > 0])

    iteration = 0
    current = sess.run(correct_predict, feed_dict={xs: adv_x, keep_prob: 1})

    while (current != target and iteration < ITER and
            len(search_domain) > 1):

        grads_target = np.zeros((1, 784), dtype=float)
        grads_other = np.zeros((1, 784), dtype=float)

        for i, grad in enumerate(grads):
            run_grad = sess.run(grad, feed_dict={xs: adv_x, keep_prob: 1})
            if i == target:
                grads_target = np.reshape(run_grad, (1, 784))
                continue

            grads_other += np.reshape(run_grad, (1, 784))

        invalid = list(set(range(784)) - search_domain)
        grads_target[0][invalid] = -theta * np.max(np.abs(grads_target))
        grads_other[0][invalid] = theta * np.max(np.abs(grads_other))

        target_sum = grads_target.reshape((1, 784)) + grads_target.reshape((784, 1))
        other_sum = grads_other.reshape((1, 784)) + grads_other.reshape((784, 1))

        if theta > 0:
            scores_mask = ((target_sum > 0) & (other_sum < 0))
        else:
            scores_mask = ((target_sum < 0) & (other_sum > 0))

        scores = scores_mask * np.abs(target_sum * other_sum)

        #np.fill_diagonal(scores, -1000)

        best = np.argmax(scores)
        if best == 0:
            break
        p1, p2 = best % 784, best // 784

        search_domain.discard(p1)
        search_domain.discard(p2)

        if theta > 0:
            adv_x[0, p1] = np.minimum(1, adv_x[0, p1] + theta)
            adv_x[0, p2] = np.minimum(1, adv_x[0, p2] + theta)
        else:
            adv_x[0, p1] = np.minimum(0, adv_x[0, p1] - theta)
            adv_x[0, p2] = np.minimum(0, adv_x[0, p2] - theta)

        current = sess.run(correct_predict, feed_dict={xs: adv_x, keep_prob: 1})
        iteration = iteration + 1

    return current, adv_x


def jsma():

    MNIST, LABEL, ANS = get_some_data(MAKE_ATTACK_NUM)

    g_sub_make = tf.Graph()
    with g_sub_make.as_default():
        xs = tf.placeholder(tf.float32, [None, 784])
        keep_prob = tf.placeholder(tf.float32)
        predict, correct_predict = cnn_structure(xs, keep_prob)
        saver = tf.train.Saver()

    sess = tf.Session(graph=g_sub_make)
    saver.restore(sess, "CNN_net/save_net.ckpt")

    gradients = []
    for i in range(10):
        derivatives, = tf.gradients(predict[:, i], xs)
        gradients.append(derivatives)

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    for i in range(10):
        if not os.path.exists('%s/%s' % (DIR, str(i))):
            os.makedirs('%s/%s' % (DIR, str(i)))

    count = 0

    for target in range(10):
        for i, image in enumerate(MNIST):
            image = np.reshape(image, (1, 784))
            test1 = sess.run(correct_predict, feed_dict={xs: image, keep_prob: 1})
            test2, adv_x = jsmaOnePicture(sess, image, xs, keep_prob, correct_predict, target, THETA, gradients)

            if target == test2:
                suc = 1
            else:
                suc = 0
            im = adv_x.reshape(28, 28)
            im = 255 * im
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im.save('%s/%s/%s_%s.bmp' % (DIR, str(target), str(count), str(suc)), 'bmp')
            count = count + 1

            print(i, test1, test2)


    #for i, d in enumerate(deltax):
    #    mean = np.mean(d)
    #    std = np.std(d)
    #    deltax[i] = (d - mean) / std

    #for i, a in enumerate(deltax):
     #   print(a)
'''
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
    jsma()

