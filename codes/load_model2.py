import tensorflow as tf
import numpy as np
import os
import time
import utils


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

    # S-LSTM
    with tf.name_scope('S_LSTM'):
        W_s1 = tf.Variable(tf.random_normal([1024, 4096]), name='W_s1')
        W_s2 = tf.Variable(tf.random_normal([40, 1024]), name='W_s2')
        W_s1_ext = tf.tile(tf.expand_dims(W_s1, 0), [num_test, 1, 1])
        W_s2_ext = tf.tile(tf.expand_dims(W_s2, 0), [num_test, 1, 1])
        cell_s = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=256)
        A = tf.math.softmax(tf.matmul(W_s2_ext, tf.math.tanh(tf.matmul(W_s1_ext, x, transpose_b=True))))
        M = tf.matmul(A, x)
        output_s_tmp, _ = tf.nn.dynamic_rnn(cell_s, inputs=M, dtype=tf.float32)
        output_s_tmp = output_s_tmp[:, -1, :]
        output_s = tf.contrib.layers.fully_connected(tf.nn.dropout(output_s_tmp, keep_prob), 64)

    # Loss and Spearman's correlation
    with tf.name_scope('loss'):
        v_cat = output_s
        y_pred = 50 * tf.contrib.layers.fully_connected(tf.nn.dropout(v_cat, keep_prob), 2, tf.math.sigmoid)
        loss = tf.reduce_mean(tf.square(y - y_pred), axis=0)

    # Sess and Restorer
    sess = tf.Session()
    restorer = tf.train.Saver()
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
    Predict('models/model2/model2.ckpt')

