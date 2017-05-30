import time
import tensorflow as tf
import numpy
import muct_input
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import convert_to_tensor
from tfobjs import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True


record_log = True
record_log_dir = './log/square_loss_relu_singlefc_conv4_k33/'

data_dir = './muct-data-bin/'


global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(1e-3, global_step, 500, 0.95, staircase=False, name='learning_rate')

[img_train, pts_train] = muct_input.inputs(False, data_dir, 128)
[img_test, pts_test] = muct_input.inputs(True, data_dir, 128)

# ========================

sess = tf.InteractiveSession(config=config)

with tf.variable_scope('input'):
    ph_x = tf.placeholder(tf.float32, [None] + img_train.shape.as_list()[1:], name='ph_img')
    ph_pts = tf.placeholder(tf.float32, [None] + pts_train.shape.as_list()[1:], name='ph_pts')
    is_training = tf.placeholder(tf.bool, name='ph_is_training')

# ===================48-->24
with tf.variable_scope('layer1'):
    layer1 = ConvObj()
    layer1.set_input(ph_x)
    layer1.batch_norm(layer1.conv2d([3, 3], 64, [1, 2, 2, 1]), is_training=is_training)
    layer1.set_output(tf.nn.relu(layer1.bn))
    
# ===================24-->12
with tf.variable_scope('layer2'):
    layer2 = ConvObj()
    layer2.set_input(layer1.output)
    layer2.batch_norm(layer2.conv2d([3, 3], 128, [1, 2, 2, 1]), is_training=is_training)
    layer2.set_output(tf.nn.relu(layer2.bn))
    
# ===================12-->6
with tf.variable_scope('layer3'):
    layer3 = ConvObj()
    layer3.set_input(layer2.output)
    layer3.batch_norm(layer3.conv2d([3, 3], 256, [1, 2, 2, 1]), is_training=is_training)
    layer3.set_output(tf.nn.relu(layer3.bn))
    
# ===================6-->3
with tf.variable_scope('layer4'):
    layer4 = ConvObj()
    layer4.set_input(layer3.output)
    layer4.batch_norm(layer4.conv2d([3, 3], 512, [1, 2, 2, 1]), is_training=is_training)
    layer4.set_output(tf.nn.relu(layer4.bn))

# ===================3*3*512-->pts
with tf.variable_scope('layer5'):
    layer5 = FcObj()
    layer5.set_input(layer4.output)
    layer5.fc(pts_train.shape.as_list()[1])
    layer5.set_output(layer5.logit)
    


with tf.variable_scope('loss'):
    weight_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cross_entropy = tf.reduce_mean(tf.square(layer5.output - ph_pts))
    total_loss = cross_entropy + weight_loss
    accuarcy = tf.reduce_mean(tf.abs(layer5.output - ph_pts))
    summary_losses = [tf.summary.scalar('cross_entropy', cross_entropy), tf.summary.scalar('weight_loss', weight_loss)]

# ===================

train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

# ===================

merged = tf.summary.merge([summary_losses])

if record_log:
    train_writer = tf.summary.FileWriter(os.path.join(record_log_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(record_log_dir, 'test'), sess.graph)

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()

saver = tf.train.Saver()

# ========restore===============
# saver.restore(sess, tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path)
# tf.train.write_graph(sess.graph_def, record_log_dir, "graph.pb", as_text = False);

# builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
# builder.add_meta_graph_and_variables(sess, ["tf-muct"])
# builder.save()
# =============================

start_time = time.time()

while(True):
    [img_, pts_] = sess.run([img_train, pts_train])
    [ _] = sess.run([train_step], feed_dict={ph_x: img_, ph_pts: pts_, is_training: True})

    if global_step.eval() % 100 == 0 :
        print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
        [img_, pts_] = sess.run([img_train, pts_train])
        [summary_train, acc_train] = sess.run([merged, accuarcy], feed_dict={ph_x: img_, ph_pts: pts_, is_training: False})
        [img_, pts_] = sess.run([img_test, pts_test])
        [summary_test, acc_test] = sess.run([merged , accuarcy], feed_dict={ph_x: img_, ph_pts: pts_, is_training: False})
        if record_log:
            train_writer.add_summary(summary_train, global_step.eval())
            test_writer.add_summary(summary_test, global_step.eval())
        print(acc_train)
        print(acc_test)
    if global_step.eval() % 500 == 0 :
        if record_log:
            saver.save(sess, os.path.join(record_log_dir, 'model.ckpt'), global_step.eval())
        pass

print('total time = ', time.time() - start_time, 's')

if record_log:
    train_writer.close();
    test_writer.close();


