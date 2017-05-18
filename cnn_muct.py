import time
import tensorflow as tf
import numpy
import muct_input
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import convert_to_tensor

record_log = False
record_log_dir = './log/newData_rough01/'

data_dir = './muct-data-bin/'

image_shape = [48, 48, 1]
point_counts = 21

conv1_layer_size = 32
conv2_layer_size = 48
conv3_layer_size = 64
final_layer_size = 6 * 6
fc1_layer_size = 64 * final_layer_size  # 3 * 3 * conv3_layer_size
fc2_layer_size = 32 * final_layer_size  # 3 * 3 * conv3_layer_size
output_layer_size = point_counts * 2


iters = int(3000000)
kp = 0.5  # keep_prob

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 500, 0.95, staircase=False)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True



class batch_norm_obj():
    bn_beta = None;
    bn_scale = None;
    bn_pop_mean = None;
    bn_pop_var = None;
    def __init__(self, shape):
        self.bn_beta = tf.Variable(tf.zeros([shape[-1]]))
        self.bn_scale = tf.Variable(tf.ones([shape[-1]]))
        self.bn_pop_mean = tf.Variable(tf.zeros([shape[-1]]), trainable=False)
        self.bn_pop_var = tf.Variable(tf.ones([shape[-1]]), trainable=False)
    
    def batch_norm(self, inputs, is_training, decay=0.999, is_conv_out=True):
        if is_training:
            if is_conv_out:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            else:
                batch_mean, batch_var = tf.nn.moments(inputs, [0])   

            train_mean = tf.assign(self.bn_pop_mean, self.bn_pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(self.bn_pop_var, self.bn_pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.bn_beta, self.bn_scale, 0.001)
        else:
            return tf.nn.batch_normalization(inputs, self.bn_pop_mean, self.bn_pop_var, self.bn_beta, self.bn_scale, 0.001)
    




[images_train, labels_train] = muct_input.inputs(False, data_dir, 128)
[images_train_probe, labels_train_probe] = muct_input.inputs(False, data_dir, 256)
[images_test_probe, labels_test_probe] = muct_input.inputs(True, data_dir, 256)





# ========================

is_training = tf.placeholder(tf.bool)

sess = tf.InteractiveSession(config=config)
x = tf.placeholder(tf.float32, shape=[None] + image_shape)

y_gt = tf.placeholder(tf.float32, shape=[None] + [point_counts * 2])
summary_input = tf.summary.image('input', x, max_outputs=16)

# x_bn_obj = batch_norm_obj(image_shape)
# x_norm = tf.cond(is_training, lambda: x_bn_obj.batch_norm(x, True), lambda: x_bn_obj.batch_norm(x, False))
x_norm = x
# =======================
output_norm_x = tf.reshape(tf.image.per_image_standardization(x[0]), [1] + image_shape)
# ===================
with tf.name_scope('layer1'):
    W_conv1 = tf.get_variable('W_conv1', [5, 5, 1, conv1_layer_size], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
    b_conv1 = tf.Variable(tf.constant(0.0, shape=[conv1_layer_size]))
    
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_norm, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # h_pool1_bn_obj = batch_norm_obj([16, 16, conv1_layer_size])
    # h_norm1 = tf.cond(is_training, lambda: h_pool1_bn_obj.batch_norm(h_pool1, True), lambda: h_pool1_bn_obj.batch_norm(h_pool1, False))
    summary_layer1 = [tf.summary.histogram('W_conv1', W_conv1), tf.summary.histogram('b_conv1', b_conv1),
                      tf.summary.histogram('h_conv1', h_conv1)]
# ===================

with tf.name_scope('layer2'):
    W_conv2 = tf.get_variable('W_conv2', [5, 5, conv1_layer_size, conv2_layer_size], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
    b_conv2 = tf.Variable(tf.constant(0.0, shape=[conv2_layer_size]))

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # h_pool2_bn_obj = batch_norm_obj([8, 8, conv2_layer_size])
    # h_norm2 = tf.cond(is_training, lambda: h_pool2_bn_obj.batch_norm(h_pool2, True), lambda: h_pool2_bn_obj.batch_norm(h_pool2, False))
    h_norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    summary_layer2 = [tf.summary.histogram('W_conv2', W_conv2), tf.summary.histogram('b_conv2', b_conv2),
                      tf.summary.histogram('h_conv2', h_conv2)]
# ===================

with tf.name_scope('layer3'):
    W_conv3 = tf.get_variable('W_conv3', [5, 5, conv2_layer_size, conv3_layer_size], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
    b_conv3 = tf.Variable(tf.constant(0.0, shape=[conv3_layer_size]))

    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_norm2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # h_pool3_bn_obj = batch_norm_obj([4, 4, conv3_layer_size])
    # h_norm3 = tf.cond(is_training, lambda: h_pool3_bn_obj.batch_norm(h_pool3, True), lambda: h_pool3_bn_obj.batch_norm(h_pool3, False))
    h_norm3 = tf.nn.lrn(h_pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    summary_layer3 = [tf.summary.histogram('W_conv3', W_conv3), tf.summary.histogram('b_conv3', b_conv3),
                      tf.summary.histogram('h_conv3', h_conv3)]
# ===================

with tf.name_scope('fc1'):
    h_norm3_flat = tf.reshape(h_norm3, [-1, final_layer_size * conv3_layer_size])
    W_fc1 = tf.get_variable('W_fc1', [final_layer_size * conv3_layer_size, fc1_layer_size], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
    b_fc1 = tf.Variable(tf.constant(0.0, shape=[fc1_layer_size]))
    h_fc1 = tf.nn.relu(tf.matmul(h_norm3_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    summary_fc1 = [tf.summary.histogram('W_fc1', W_fc1), tf.summary.histogram('b_fc1', b_fc1),
                      tf.summary.histogram('h_fc1', h_fc1)]
# ===================

with tf.name_scope('fc2'):
    W_fc2 = tf.get_variable('W_fc2', [fc1_layer_size, fc2_layer_size], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))
    b_fc2 = tf.Variable(tf.constant(0.0, shape=[fc2_layer_size]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    summary_fc2 = [tf.summary.histogram('W_fc2', W_fc2), tf.summary.histogram('b_fc2', b_fc2),
                      tf.summary.histogram('h_fc2', h_fc2)]
# ===================

with tf.name_scope('fc_final'):
    W_final = tf.get_variable('W_final', [fc2_layer_size, output_layer_size], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

    b_final = tf.Variable(tf.constant(0.0, shape=[output_layer_size]))
    logits = tf.matmul(h_fc2, W_final) + b_final
    summary_fc_final = [tf.summary.histogram('W_final', W_final), tf.summary.histogram('b_final', b_final),
                      tf.summary.histogram('logits', logits)]
# ===================

with tf.name_scope('loss'):
    weight_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cross_entropy = tf.reduce_mean(tf.square(tf.multiply(tf.subtract(logits, y_gt), 1)))
    total_loss = cross_entropy + weight_loss
    accuracy = tf.reduce_mean(tf.square(tf.subtract(logits, y_gt)))
    summary_losses = [tf.summary.scalar('cross_entropy', cross_entropy), tf.summary.scalar('weight_loss', weight_loss),
                      tf.summary.scalar('accuracy', accuracy)]

# ===================

train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

# ===================

merged_heavy = tf.summary.merge([summary_losses, summary_input, summary_layer1, summary_layer2, summary_layer3, summary_fc1, summary_fc2, summary_fc_final])
merged_light = tf.summary.merge([summary_layer1, summary_layer2, summary_layer3, summary_fc1, summary_fc2, summary_fc_final])

if record_log:
    train_writer = tf.summary.FileWriter(os.path.join(record_log_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(record_log_dir, 'test'), sess.graph)

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()

saver = tf.train.Saver()

# ========restore===============
saver.restore(sess, os.path.join(record_log_dir, "model.ckpt-4000"))
learning_rate = tf.train.exponential_decay(1e-4, global_step, 500, 0.95, staircase=False)
# tf.train.write_graph(sess.graph_def, "./log/", "graph.pb", as_text=True);

# builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
# builder.add_meta_graph_and_variables(sess, ["tf-muct"])
# builder.save()
# =============================

start_time = time.time()

for i in range(iters):
    [image_batch, label_batch] = sess.run([images_train, labels_train])
    [train_summary, _] = sess.run([merged_light, train_step], feed_dict={x: image_batch, y_gt: label_batch, keep_prob: kp, is_training: True})
    if record_log:
        train_writer.add_summary(train_summary, global_step.eval())
    # sess.run(train_step, feed_dict={x: image_batch, y_gt: label_batch, keep_prob: kp, is_training: True})
    if i % 100 == 0 :
        print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
        [image_train_probe, label_train_probe] = sess.run([images_train_probe, labels_train_probe])
        [image_test_probe, label_test_probe] = sess.run([images_test_probe, labels_test_probe])
        [train_summary, train_accuracy] = sess.run([merged_heavy, accuracy], feed_dict={x: image_train_probe, y_gt: label_train_probe, keep_prob: 1.0, is_training: False})
        [test_summary, test_accuracy] = sess.run([merged_heavy, accuracy], feed_dict={x: image_test_probe, y_gt: label_test_probe, keep_prob: 1.0, is_training: False})
        if record_log:
            train_writer.add_summary(train_summary, global_step.eval())
            run_meta = tf.RunMetadata();
            train_writer.add_run_metadata(run_meta, 'step%d' % global_step.eval(), global_step=global_step.eval())
            test_writer.add_summary(test_summary, global_step.eval())
        print('test/train accuracy = %g/%g' % (test_accuracy, train_accuracy))
    if i % 1000 == 999 :
        if record_log:
            saver.save(sess, os.path.join(record_log_dir, 'model.ckpt'), global_step.eval())
        pass

print('total time = ', time.time() - start_time, 's')

if record_log:
    train_writer.close();
    test_writer.close();


