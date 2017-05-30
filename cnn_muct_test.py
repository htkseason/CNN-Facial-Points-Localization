import os
import tensorflow as tf

import numpy
import matplotlib.pyplot as plt
import muct_input
from tensorflow.python.platform import gfile
from celeba import celeba_input

record_log_dir = './log/square_loss_relu_singlefc_conv4/'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True


sess = tf.InteractiveSession(config=config)

test_image = tf.image.resize_images(tf.image.rgb_to_grayscale(celeba_input.inputs(128)), [48, 48])
# [test_image, _] = muct_input.inputs(True, './muct-data-bin/', 128)


#===restore
saver = tf.train.import_meta_graph(tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path + '.meta')
saver.restore(sess, tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path)
x = tf.get_default_graph().get_tensor_by_name('input/ph_img:0')
output = tf.get_default_graph().get_tensor_by_name('layer5/output:0')
is_training = tf.get_default_graph().get_tensor_by_name('input/ph_is_training:0')


#====testing
tf.train.start_queue_runners()

imgs = test_image.eval()
[pts] = sess.run([output], feed_dict={x:imgs , is_training : False})

for i in range(imgs.shape[0]):
    pic = numpy.zeros([96, 96, 1])
    pic[24:24 + 48, 24:24 + 48, :] = imgs[i, :, :, :]; 
    lab = pts[i] + 24

    for p in range(int(lab.shape[0] / 2)):
        if (lab[p * 2 + 1] >= 96 or lab[p * 2 + 1] < 0 or lab[p * 2] >= 96 or lab[p * 2 ] < 0):
            pass
        else:
            pic[int(lab[p * 2 + 1]), int(lab[p * 2]), 0] = 5;
            pass
        
    plt.imshow(pic[:, :, 0], cmap='gray')
    plt.show()
pass
