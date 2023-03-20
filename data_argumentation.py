import os
import keras
# import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import tensorflow as tf

#
# def creat_x_database(rootdir, resize_row, resize_col):
#     list = os.listdir(rootdir)
#     database = np.arange(len(list) * resize_row * resize_col * 3).reshape(len(list), resize_row, resize_col, 3)
#     for i in range(0, len(list)):
#         path = os.path.join(rootdir, list[i])  # 把目录和文件名合成一个路径
#         if os.path.isfile(path):  ##判断路径是否为文件
#             image_raw_data = tf.gfile.FastGFile(path, 'rb').read()
#             with tf.Session() as sess:
#                 img_data = tf.image.decode_jpeg(image_raw_data)
#                 resized = tf.image.resize_images(img_data, [resize_row, resize_col], method=0)
#                 database[i] = resized.eval()
#     return database


# '''
# 用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，
# 直到达到规定的epoch次数为止。
# '''
datagen = ImageDataGenerator(
    featurewise_center=True,  # 去中心化
    featurewise_std_normalization=True,  # 标准化
    rotation_range=45,  # 旋转范围, 随机旋转(0-180)度
    width_shift_range=0.2,  # 随机沿着水平或者垂直方向，以图像的长宽小部分百分比为变化范围进行平移;
    height_shift_range=0.2,
    shear_range=0.2,  # 水平或垂直投影变换
    zoom_range=0.2,  # 按比例随机缩放图像尺寸
    horizontal_flip=True,  # 水平翻转图像
    rescale=1,
    fill_mode='nearest')  # 填充像素, 出现在旋转或平移之后

# img=load_img('E:/data/muck truck v2/fail/FFCV07001_0_20170304195758240.jpg')
# x=img_to_array(img)
# x=x.reshape((1,)+x.shape) # this is a Numpy array with shape (1, 3, 150, 150)

# x = creat_x_database('E:/data/muck truck v4 muticlass/qualified', 256, 256)
# datagen.fit(x)
# x = creat_x_database('E:/TCN/photos', 1000, 1000)
# datagen.fit(x)

i = 0
for batch in datagen.flow_from_directory(directory='E:/TCN/photos',
                                         target_size=(299, 299), color_mode='rgb',
                                         classes=None, class_mode='categorical',
                                         batch_size=32, shuffle=True, seed=None,
                                         save_to_dir='E:/TCN/target',
                                         save_prefix='qualified',
                                         save_format='png',
                                         follow_links=False):
    i +=1
    if i > 100:
        break


    # def flow_from_directory(self, directory,
    #                         target_size=(256, 256), color_mode='rgb',
    #                         classes=None, class_mode='categorical',
    #                         batch_size=32, shuffle=True, seed=None,
    #                         save_to_dir=None,
    #                         save_prefix='',
    #                         save_format='png',
    #                         follow_links=False):
    #     return DirectoryIterator(
    #         directory, self,
    #         target_size=target_size, color_mode=color_mode,
    #         classes=classes, class_mode=class_mode,
    #         data_format=self.data_format,
    #         batch_size=batch_size, shuffle=shuffle, seed=seed,
    #         save_to_dir=save_to_dir,
    #         save_prefix=save_prefix,
    #         save_format=save_format,
    #         follow_links=follow_links)