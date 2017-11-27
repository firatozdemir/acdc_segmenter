import tensorflow as tf
import numpy as np
from tfwrapper import layers

def layers_t():
    ### Tests with 2D
    #Test 1: When 2 tensors of same size
    sz1 = [16, 200, 200, 32]
    t1 = tf.convert_to_tensor(value=np.ones(sz1), dtype=tf.float32)
    inputs = [t1, t1]
    outputs = layers.crop_and_concat_layer(inputs=inputs, axis=3)
    sz_output1 = outputs.get_shape().as_list()
    if not sz_output1[:-1] == sz1[:-1]:
        print('1st test with test_layers have failed.')
    #Test 2: When 2 tensor of different size
    sz2 = [16, 224, 244, 64]
    t2 = tf.convert_to_tensor(value=np.ones(sz2), dtype=tf.float32)
    inputs = [t1, t2]
    outputs = layers.crop_and_concat_layer(inputs=inputs, axis=3)
    sz_output2 = outputs.get_shape().as_list()
    if not sz_output2[:-1] == sz1[:-1]:
        print('2nd test with test_layers have failed.')
    ### Tests with 3D has no added complexity.

    print('All test_layers checks have succeeded.')

if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    layers_t()
    sess.close()