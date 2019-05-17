import tensorflow as tf
import numpy as np
import os
import time
import utils
from rnn_cells.skip_rnn_cells import SkipLSTMCell


def Predict(model=None):

    # Read data_set from txt file
    test_set = np.loadtxt(os.path.join('data', 'test_dataset.txt'), dtype=np.float32)
    feature_dir = os.path.join('data', 'c3d_feat')
    test_index = test_set[:, 0]
    test_y = test_set[:, 1:3]
    num_test = 50   # feed half of the test set one time

    # Print pid
    print('pid:' + str(os.getpid()))

    # Model definition
    x = tf.placeholder(tf.float32, [num_test, None, 4096], name='inputs')
    y = tf.placeholder(tf.float32, [None, 2], name='scores')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # M-LSTM
    with tf.name_scope('M_LSTM'):
        cell_m1 = tf.nn.rnn_cell.LSTMCell(num_units=256, name='M_LSTM_1')
        cell_m2 = tf.nn.rnn_cell.LSTMCell(num_units=256, name='M_LSTM_2')
        cell_m3 = tf.nn.rnn_cell.LSTMCell(num_units=128, name='M_LSTM_3')
        x_ext = tf.expand_dims(x, -1)
        conv_1 = tf.reduce_mean(tf.layers.conv2d(x_ext, 1, [8, 1], [2, 1], "VALID"), -1)
        conv_2 = tf.reduce_mean(tf.layers.conv2d(x_ext, 1, [4, 1], [2, 1], "VALID"), -1)
        conv_3 = tf.reduce_mean(tf.layers.conv2d(x_ext, 1, [1, 1], [1, 1], "VALID"), -1)

        W_m11 = tf.Variable(tf.random_normal([512, 4096]), name='W_m11')
        W_m12 = tf.Variable(tf.random_normal([40, 512]), name='W_m12')
        W_m11_ext = tf.tile(tf.expand_dims(W_m11, 0), [num_test, 1, 1])
        W_m12_ext = tf.tile(tf.expand_dims(W_m12, 0), [num_test, 1, 1])
        A_m1 = tf.math.softmax(tf.matmul(W_m12_ext, tf.math.tanh(tf.matmul(W_m11_ext, conv_1, transpose_b=True))))
        M_m1 = tf.matmul(A_m1, conv_1)

        W_m21 = tf.Variable(tf.random_normal([512, 4096]), name='W_m21')
        W_m22 = tf.Variable(tf.random_normal([40, 512]), name='W_m22')
        W_m21_ext = tf.tile(tf.expand_dims(W_m21, 0), [num_test, 1, 1])
        W_m22_ext = tf.tile(tf.expand_dims(W_m22, 0), [num_test, 1, 1])
        A_m2 = tf.math.softmax(tf.matmul(W_m22_ext, tf.math.tanh(tf.matmul(W_m21_ext, conv_2, transpose_b=True))))
        M_m2 = tf.matmul(A_m2, conv_2)

        W_m31 = tf.Variable(tf.random_normal([512, 4096]), name='W_m31')
        W_m32 = tf.Variable(tf.random_normal([40, 512]), name='W_m32')
        W_m31_ext = tf.tile(tf.expand_dims(W_m31, 0), [num_test, 1, 1])
        W_m32_ext = tf.tile(tf.expand_dims(W_m32, 0), [num_test, 1, 1])
        A_m3 = tf.math.softmax(tf.matmul(W_m32_ext, tf.math.tanh(tf.matmul(W_m31_ext, conv_3, transpose_b=True))))
        M_m3 = tf.matmul(A_m3, conv_3)

        output_m1_tmp, _ = tf.nn.dynamic_rnn(cell=cell_m1, inputs=M_m1, dtype=tf.float32)
        output_m2_tmp, _ = tf.nn.dynamic_rnn(cell=cell_m2, inputs=M_m2, dtype=tf.float32)
        output_m3_tmp, _ = tf.nn.dynamic_rnn(cell=cell_m3, inputs=M_m3, dtype=tf.float32)
        output_m1_tmp = output_m1_tmp[:, -1, :]  # take out the last output
        output_m2_tmp = output_m2_tmp[:, -1, :]  # take out the last output
        output_m3_tmp = output_m3_tmp[:, -1, :]  # take out the last output
        output_m1 = tf.contrib.layers.fully_connected(tf.nn.dropout(output_m1_tmp, keep_prob), 64)
        output_m2 = tf.contrib.layers.fully_connected(tf.nn.dropout(output_m2_tmp, keep_prob), 64)
        output_m3 = tf.contrib.layers.fully_connected(tf.nn.dropout(output_m3_tmp, keep_prob), 64)

    # Loss and Spearman's correlation
    with tf.name_scope('loss'):
        v_cat = tf.concat([output_m1, output_m2, output_m3], -1)
        y_fc = tf.contrib.layers.fully_connected(tf.nn.dropout(v_cat, keep_prob), 64)
        y_pred = 50 * tf.contrib.layers.fully_connected(y_fc, 2, tf.sigmoid)
        loss = tf.reduce_mean(tf.square(y - y_pred), axis=0)

    # Sess and Restorer
    sess = tf.Session()
    restorer = tf.train.Saver(tf.global_variables())
    restorer.restore(sess, model)

    # Initialization
    start_time = time.time()
    test_x = utils.get_feature_batch(test_index, feature_dir)

    # compute and display
    MSE1, y_pred1 = sess.run([loss, y_pred], feed_dict={keep_prob: 1.0, x: test_x[0:50, :, :], y: test_y[0:50, :]})
    MSE2, y_pred2 = sess.run([loss, y_pred], feed_dict={keep_prob: 1.0, x: test_x[50:, :, :], y: test_y[50:, :]})
    MSE = (MSE1 + MSE2) / 2
    y_pre = tf.concat([y_pred1, y_pred2], 0)
    _, pearson_r1 = tf.contrib.metrics.streaming_pearson_correlation(y_pre[:, 0], test_y[:, 0])
    _, pearson_r2 = tf.contrib.metrics.streaming_pearson_correlation(y_pre[:, 1], test_y[:, 1])
    sess.run(tf.local_variables_initializer())
    r = sess.run([pearson_r1, pearson_r2])
    end_time = time.time()
    print('Test: \n MSE of TES: {:.7f} MSE of PCS: {:.7f} \n corr of TES: {:.7f} corr of PCS: {:.7f} \n PID: {} Elapsed time: {}'
          .format(MSE[0], MSE[1], r[0], r[1], os.getpid(), utils.format_time(end_time - start_time)))
    sess.close()


if __name__ == '__main__':
    Predict('models/model1/model1.ckpt')

