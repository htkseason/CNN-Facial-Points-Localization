import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import muct_input

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

[images, labels] = muct_input.inputs(False, './muct-data-bin/', 128)

tf.train.start_queue_runners()
shape = [48,48,1]
[pics, labs] = sess.run([images, labels])
print(images.shape)
print(labels.shape)
for i in range(images.shape[0]):
    pic = pics[i]
    lab = labs[i]
    print(lab)
    for p in range(int(lab.shape[0] / 2)):
        if (lab[p * 2 + 1] >= pic.shape[0] or lab[p * 2 + 1] < 0 or lab[p * 2] >= pic.shape[1] or lab[p * 2 ] < 0):
            pass
        else:
            pic[int(lab[p * 2 + 1]), int(lab[p * 2]), 0] = 5;
            pass
        
    plt.imshow(pic[:, :, 0], cmap='gray')
    plt.show()
pass
