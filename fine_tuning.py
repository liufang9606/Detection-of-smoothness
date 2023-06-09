# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:41:14 2019
Inception_v3迁移学习,将预训练的inception-v3的网络参数在flower_photo中进行训练
再进行测试，得到测试结果
@author: zhaoy
"""

# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 加载通过TensorFlow-Slim定义好的inception_v3模型。
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 处理好之后的数据文件。
INPUT_DATA = 'my_processed_data.npy'
# 保存训练好的模型的路径。
TRAIN_FILE = './model/'

# 在运行时需要先自行从Google下载inception_v3.ckpt文件存储路径，本项目已将模型其保存在当前路径下。
CKPT_FILE = 'inception_v3.ckpt'

# 定义训练中使用的参数。
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 2

# 不需要从谷歌训练好的模型中加载的参数，Auxiliary Logits层输出默认为1000，在迁移时需要对这一层进行修改
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数名称，在fine-tuning的过程中就是最后的全联接层。
TRAINABLE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogits'


# 获取所有需要从谷歌训练好的模型中加载的参数。
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    # 获取所有需要从inception-v3模型中加载的参数，
    #也即是除logits和auxlogits层意外的参数都放入variables_to_restore列表中
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            #检查字符串是否已制定的字符串开头，如果是则返回True,否则返回False
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# 获取所有需要训练的变量列表。
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_trian = []

    # 枚举所有需要训练的参数前缀scope，并通过这些前缀找到所有需要训练的参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_trian.extend(variables)
    return variables_to_trian

tf.reset_default_graph()
def main():
    # 加载预处理好的数据。
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]

    validation_images = processed_data[2]
    validation_labels = processed_data[3]

    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    print("%d training examples, %d validation examples and %d testing examples." % (
        n_training_example, len(validation_labels), len(testing_labels)))

    # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # 定义inception-v3模型。因为谷歌给出的只有模型参数取值，所以这里
    # 需要在这个代码中定义inception-v3的模型结构。虽然理论上需要区分训练和
    # 测试中使用到的模型，也就是说在测试时应该使用is_training=False，但是
    # 因为预先训练好的inception-v3模型中使用的batch normalization参数与
    # 新的数据会有出入，所以这里直接使用同一个模型来做测试。
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES, is_training=True)
    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()
    # 定义交叉熵损失，在模型定义的时候已经将正则化损失加入损失集合了。
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    # 定义训练过程。这里minimize的过程中指定了需要优化的变量集合。
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.arg_max(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义加载Google训练好的Inception-v3模型的Saver。
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True)

    # 定义保存新模型的Saver。
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化没有加载的变量
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载Google已经训练好的模型
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS+1):
           sess.run(train_step, feed_dict={
               images: training_images[start: end],
               labels: training_labels[start: end]})

           if i % 30 == 0 or i == STEPS:
               saver.save(sess, TRAIN_FILE, global_step=i)
               print(np.sum(training_labels[start: end]))

               validation_accuracy = sess.run(evaluation_step, feed_dict={
                   images: validation_images, labels: validation_labels})
               print('Step %d: Validation accuracy = %.1f%%' % (
                   i, validation_accuracy * 100.0))

           start = end
           if start == n_training_example:
               start = 0
           end = start + BATCH
           if end > n_training_example:
               end = n_training_example

        # 在最后的测试数据上测试正确率。
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()
