from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 203210
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 39402

min_fraction_of_examples_in_queue = 0.1


def read_record(image_file_queue, label_file_queue):

    class MuctRecord(object):
        pass

  
    result = MuctRecord()
    height = 48
    width = 48
    depth = 1
    points = 21
  
    image_bytes = height * width * depth * 1;
    label_bytes = points * 2 * 2;  # pts_count*2(x,y)*2(16U) 
  
    image_reader = tf.FixedLengthRecordReader(record_bytes=image_bytes)
    _, value = image_reader.read(image_file_queue)
    result.image = tf.decode_raw(value, tf.uint8)
    result.image = tf.reshape(result.image, [depth, height, width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.image = tf.transpose(result.image, [1, 2, 0])
    result.image = tf.cast(result.image, tf.float32)
  
    label_reader = tf.FixedLengthRecordReader(record_bytes=label_bytes)
    _, value = label_reader.read(label_file_queue)
    result.label = tf.decode_raw(value, tf.int16)
    result.label = tf.cast(result.label, tf.float32)
    result.label = tf.reshape(result.label, [points * 2])
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return image_batch, label_batch


def inputs(eval_data, data_dir, batch_size):
  
    if not eval_data:
        image_filenames = [os.path.join(data_dir, 'muct_image_train_%d' % i) for i in range(0, 5)]
        label_filenames = [os.path.join(data_dir, 'muct_points_train')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        image_filenames = [os.path.join(data_dir, 'muct_image_test')]
        label_filenames = [os.path.join(data_dir, 'muct_points_test')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in image_filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    for f in label_filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

  
    image_filename_queue = tf.train.string_input_producer(image_filenames, shuffle=False)
    label_filename_queue = tf.train.string_input_producer(label_filenames, shuffle=False)

  
    record = read_record(image_filename_queue, label_filename_queue)
    # record.image = tf.image.resize_images(record.image, [48, 48])
    # record.label = tf.multiply(record.label, 48.0 / 94.0)
    record.image = tf.image.per_image_standardization(record.image)


    min_queue_examples = int(num_examples_per_epoch * 
                           min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(record.image, record.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)



