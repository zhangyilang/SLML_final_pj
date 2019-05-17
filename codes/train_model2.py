import sys
import os
import time
import random
import tensorflow as tf
import numpy as np
import argparse
import utils


def main(args):

    # Read data_set from txt file
    train_set = np.loadtxt(os.path.join(args.data_set, 'train_dataset.txt'))
    train_feature = os.path.join(args.data_set, 'c3d_feat')
    train_index = train_set[:, 0]
    train_y = train_set[:, 1:3]

    # Log and model directory.
    log_dir = os.path.join(args.log_dir, args.model_name)
    if not os.path.isdir(log_dir):  # Create log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(args.model_dir, args.model_name)
    if not os.path.isdir(model_dir):    # Create model directory if it doesn't exist
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, args.model_name + '.ckpt')
    max_checkpoints = 3

    # Print pid
    print('pid:' + str(os.getpid()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # Model definition
    x = tf.placeholder(tf.float32, [args.batch_size, None, 4096], name='inputs')
    y = tf.placeholder(tf.float32, [None, 2], name='scores')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    # S-LSTM
    with tf.name_scope('S_LSTM'):
        W_s1 = tf.Variable(tf.random_normal([1024, 4096]), name='W_s1')
        W_s2 = tf.Variable(tf.random_normal([40, 1024]), name='W_s2')
        W_s1_ext = tf.tile(tf.expand_dims(W_s1, 0), [args.batch_size, 1, 1])
        W_s2_ext = tf.tile(tf.expand_dims(W_s2, 0), [args.batch_size, 1, 1])
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
        penalty = tf.nn.l2_loss(tf.matmul(A, A, transpose_b=True)-tf.eye(40, batch_shape=[args.batch_size])) * 2
        loss = tf.reduce_mean(tf.square(y - y_pred), axis=0) + 0.001 * penalty
        _, pearson_r1 = tf.contrib.metrics.streaming_pearson_correlation(y_pred[:, 0], y[:, 0])
        _, pearson_r2 = tf.contrib.metrics.streaming_pearson_correlation(y_pred[:, 1], y[:, 1])

    # Optimizer and gradient
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    global_step = tf.Variable(0, trainable=False)
    gradients = opt.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    trainer = opt.apply_gradients(capped_gradients, global_step=global_step)

    # Checkpoint restorer and saver
    saver = tf.train.Saver(max_to_keep=max_checkpoints)

    # Configuration about GPU
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Initialization
        sess.run(tf.initializers.global_variables())
        sess.run(tf.initializers.local_variables())
        max_step = len(train_index) * args.epoch / args.batch_size
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        start_time = time.time()
        step = sess.run(global_step)
        display_step = 10
        store_step = 100
        restorer = tf.train.Saver()
        restorer.restore(sess, 'models/model2/model2.ckpt')

        while step <= max_step:
            mask = random.sample(range(len(train_index)), args.batch_size)
            train_batch_x = utils.get_feature_batch(train_index[mask], train_feature)
            train_batch_y = train_y[mask, :]

            _, step = sess.run([trainer, global_step], feed_dict={keep_prob: args.keep_prob, lr: args.learning_rate,
                                                                  x: train_batch_x, y: train_batch_y})

            # Display
            if step % display_step == 0:
                MSE, corr1, corr2 = sess.run([loss, pearson_r1, pearson_r2], feed_dict={keep_prob: args.keep_prob,
                                             lr: args.learning_rate, x: train_batch_x, y: train_batch_y})
                int_time = time.time()
                print(
                    'Step: {:06d} --- MSE of TES: {:.7f} MSE of PCS: {:.7f} corr of TES: {:.7f} corr of PCS: {:.7f} PID: {}  Elapsed time: {}'
                    .format(step, MSE[0], MSE[1], corr1, corr2, os.getpid(), utils.format_time(int_time - start_time)))

                # Store
                if step % store_step == 0:
                    print('======================================')
                    int_time = time.time()
                    print('Elapsed time: {}'.format(utils.format_time(int_time - start_time)))
                    print('MSE: {} Pearson\'s correlation:{}'.format(MSE, [corr1, corr2]))
                    # save weights to file
                    save_path = saver.save(sess, model_path)
                    print('Variables saved in file: %s' % save_path)
                    print('Logs saved in dir: %s' % log_dir)
                    summary = tf.Summary()
                    summary.value.add(tag='loss/TES', simple_value=MSE[0])
                    summary.value.add(tag='loss/PCS', simple_value=MSE[1])
                    summary.value.add(tag='corr/TES', simple_value=corr1)
                    summary.value.add(tag='corr/PCS', simple_value=corr2)
                    train_writer.add_summary(summary, step)
                    print('======================================')

        end_time = time.time()
        print('Elapsed time: {}'.format(utils.format_time(end_time - start_time)))
        save_path = saver.save(sess, model_path)
        print('Variables saved in file: %s' % save_path)
        print('Logs saved in dir: %s' % log_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=int, help='learning rate', default=0.0001)
    parser.add_argument('--epoch', type=int, help='Number of training epochs.', default=256)
    parser.add_argument('--batch_size', type=int, help='batch size.', default=32)
    parser.add_argument('--keep_prob', type=float, help="Keep probability.", default=0.5)
    parser.add_argument('--embedding_size', type=int, help="embedding size.", default=0)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, help='CUDA VISIBLE DEVICES', default='0')
    parser.add_argument('--model_name', type=str, help='', default='model2')
    parser.add_argument('--log_dir', type=str, help='Log directory.', default='logs')
    parser.add_argument('--model_dir', type=str, help='Trained models and checkpoints.', default='models')
    parser.add_argument('--data_set', type=str, help="Data set.", default='data')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

