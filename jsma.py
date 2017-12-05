from Setting.attack_setting import *
import struct
import os
from open_data import get_some_data
import numpy as np
import tensorflow as tf
from PIL import Image
from Structure import cnn_structure, fc_structure, sub_structure
import matplotlib.pyplot as plt
if DIR == 'ADV_FC':
    from FCpredict import compute_accuracy, predict_from_array, predict_from_dir_to_dir
if DIR == 'ADV_CNN':
    from CNNpredict import compute_accuracy, predict_from_array, predict_from_dir_to_dir

def jsmaOnePicture(predict, num, a, sess, image, xs, keep_prob, correct_predict, target, theta, grads):
    
    adv_x = np.copy(image)
    add_x = np.zeros((1, 784), dtype=float)

    search_domain1 = set([i for i in range(784)
                            if adv_x[0, i] < 255])
    search_domain2 = set([i for i in range(784)
                            if adv_x[0, i] > 0])
    iteration = 0
    current = sess.run(correct_predict, feed_dict={xs: adv_x, keep_prob: 1})

    while (current != target and iteration < ITER):
        if theta > 0 and len(search_domain1) < 1:
            break
        if theta < 0 and len(search_domain2) < 1:
            break

        grads_target = np.zeros((1, 784), dtype=float)
        grads_other = np.zeros((1, 784), dtype=float)
        side_add = np.zeros((1, 784), dtype=float)

        for i, grad in enumerate(grads):
            run_grad = sess.run(grad, feed_dict={xs: adv_x, keep_prob: 1})
            if i == target:
                grads_target = np.reshape(run_grad, (1, 784))
                continue
            grads_other += np.reshape(run_grad, (1, 784))
        
        if theta > 0:
            invalid = list(set(range(784)) - search_domain1)
        else:
            invalid = list(set(range(784)) - search_domain2)

        grads_target[0][invalid] = -theta * np.max(np.abs(grads_target))
        grads_other[0][invalid] = theta * np.max(np.abs(grads_other))

        if theta > 0:
            scores_mask = ((grads_target > 0) & (grads_other < 0))
        else:
            scores_mask = ((grads_target < 0) & (grads_other > 0))
        
        if SIDE == True:
            for i in range(784):
                sum = 0
                p1, p2 = i // 28, i % 28
                tmp1l, tmp1r = p1+1, p2
                tmp2l, tmp2r = p1-1, p2
                tmp3l, tmp3r = p1, p2+1
                tmp4l, tmp4r = p1, p2-1
                if tmp1l < 28:
                    sum += adv_x[0][tmp1l*28+tmp1r]
                if tmp2l >= 0:
                    sum += adv_x[0][tmp2l*28+tmp2r]
                if tmp3r < 28:
                    sum += adv_x[0][tmp3l*28+tmp3r]
                if tmp4r >= 0:
                    sum += adv_x[0][tmp4l*28+tmp4r]

                if scores_mask[0][i] > 0:
                    side_add[0][i] = sum
                else:
                    side_add[0][i] = 0
        '''
        for i in range(784):
            if scores_mask[0][i]:
                if side_add[0][i] == 1:
                    if theta > 0:
                        adv_x[0, i] = np.minimum(1, adv_x[0, i] + theta)
                        add_x[0, i] += theta
                    else:
                        adv_x[0, i] = np.maximum(0, adv_x[0, i] + theta)
                        add_x[0, i] += theta
        
        for i in range(784):
            if adv_x[0][i] == 1:
                search_domain1.discard(i)
            if adv_x[0][i] == 0:
                search_domain2.discard(i)
        '''
        
        scores = scores_mask * (np.abs(grads_target * grads_other))
        scores += side_add * 2

        p = np.argmax(scores)
        #print(p)
        if theta > 0:
            adv_x[0, p] = np.minimum(1, adv_x[0, p] + theta)
        else:
            adv_x[0, p] = np.maximum(0, adv_x[0, p] + theta)
        
        if theta > 0:
            search_domain1.discard(p)
        else:
            search_domain2.discard(p)

        #print(sess.run(predict, feed_dict={xs: adv_x, keep_prob: 1}))
        current = sess.run(correct_predict, feed_dict={xs: adv_x, keep_prob: 1})
        iteration = iteration + 1
        theta = -theta

    return current, adv_x, iteration


def jsma():

    MNIST, LABEL, ANS = get_some_data(MAKE_ATTACK_NUM)

    g_sub_make = tf.Graph()
    with g_sub_make.as_default():
        xs = tf.placeholder(tf.float32, [None, 784])
        keep_prob = tf.placeholder(tf.float32)
        if DIR == 'ADV_FC':
            predict, correct_predict = fc_structure(xs, keep_prob)
        if DIR == 'ADV_CNN':
            predict, correct_predict = cnn_structure(xs, keep_prob)
        saver = tf.train.Saver()

    sess = tf.Session(graph=g_sub_make)
    if DIR == 'ADV_FC':
        saver.restore(sess, "FC_net/save_net.ckpt")
    if DIR == 'ADV_CNN':
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

    f, a = plt.subplots(10, 10, figsize=(12, 4))

    suc = 0
    reali = 0
    totaliter = 0
    for target in range(10):
        suc_pic = 0
        for i, image in enumerate(MNIST):
            image = np.reshape(image, (1, 784))
            test1 = sess.run(correct_predict, feed_dict={xs: image, keep_prob: 1})
            if test1 == target:
                continue
            
            test2, adv_x, iter2 = jsmaOnePicture(predict, reali, a, sess, image, xs, keep_prob, correct_predict, target, THETA, gradients)
            reali += 1
            totaliter += iter2
            '''
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
            '''
            if target == test2:
                suc += 1
                if suc_pic < 10:
                    a[target][suc_pic].imshow(np.reshape(adv_x, (28, 28)), plt.cm.gray)
                    a[target][suc_pic].set_xticks([])
                    a[target][suc_pic].set_yticks([])
                    suc_pic += 1

            print(i, test1, test2)

    print(suc / reali)
    print(totaliter / reali)
    plt.show()

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

