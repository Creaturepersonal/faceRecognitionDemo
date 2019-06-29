import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras import layers, optimizers, datasets, Sequential

batchsz = 128

#对数据元进行处理
def preprocess322(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 128, 128, 1])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=20)
    return x, y

trainimages = []
trainlabels = []
testimages = []
testlabels = []
# TrainPath = 'D:\Comprehensive3\FaceRecognition\DataSet2-FaceOfStar\\train'
# TestPath = 'D:\Comprehensive3\FaceRecognition\DataSet2-FaceOfStar\\test'


# TrainPath = 'D:\Comprehensive3\FaceRecognition\DataSet1-FaceOfAllIsWell\\train'
# TestPath = 'D:\Comprehensive3\FaceRecognition\DataSet1-FaceOfAllIsWell\\test'
#训练集图片路径和测试集图片路径
TrainPath = 'D:\Comprehensive3\FaceRecognition\FaceLib\\train'
TestPath = 'D:\Comprehensive3\FaceRecognition\FaceLib\\test'

#根据路径读取图片
def read_path322(pathname, images, labels):
    i = 0
    for dir_item in os.listdir(pathname):
        i += 1
        full_path = os.path.abspath(os.path.join(pathname, dir_item))
        if os.path.isdir(full_path):
            read_path322(full_path, images, labels)
        else:
            if dir_item.endswith('.bmp'):
                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.equalizeHist(image)
                image = cv2.resize(image, (128, 128))
                images.append(image)
                label = dir_item.split('_')[0]
                label = int(label)
                labels.append(label)
    return images, labels

#将图片转换为numpy的数组
def load_dataset322(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

#训练模型函数
def train322(trainx, trainy, testx, testy):
    conv_layers = [
        # layers.Conv2D(8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 卷积核数量，卷积核大小，填充方式，激活函数
        # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # layers.Dropout(0.5),
        #
        #
        # layers.Conv2D(16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # strides代表步长
        # layers.Dropout(0.5),
        #
        #
        # layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # layers.Dropout(0.5),
        #设置卷积和池化层数，加入dropout层是防止过拟合
        layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 卷积核数量，卷积核大小，填充方式，激活函数
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Dropout(0.25),

        layers.Conv2D(32, kernel_size=[3,3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Dropout(0.5),

        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Dropout(0.5),

        # 扁平化
        tf.keras.layers.Flatten(),

        layers.Dense(1024, activation=tf.nn.relu),  # 1024代表神经元的数量
        layers.Dense(20),#根据所分类的类别数来确定参数的值

    ]

    net = Sequential(conv_layers)
    # 输入数据为(x, 28, 28, 1)
    # 构建
    net.build(input_shape=(None, 128, 128, 1))
    net.summary()

    # 设置模型训练方法
    net.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # 训练
    net.fit(trainx, trainy, batch_size=100, epochs=15,  verbose=1, callbacks=[],
            validation_data=(testx, testy), shuffle=True)
    # 对模型进行评估
    net.evaluate(testx, testy)
    # 保存模型
    net.save('model/model.h5')
    # 保存参数
    net.save_weights('model/weights.ckpt')

#没用到
def AddTesttoTrain322(trainimages, trainlabels, testimages, testlabels):
    for i in range(len(testimages)):
        trainimages.append(testimages[i])
        trainlabels.append(testlabels[i])
    return trainimages, trainlabels, testimages, testlabels


def main():
    global TrainPath
    global TestPath
    global trainimages
    global trainlabels
    global testimages
    global testlabels
    global batchsz
    # 读取训练集图片
    trainimages, trainlabels = read_path322(TrainPath, trainimages, trainlabels)
    # 读取测试集图片
    testimages, testlabels = read_path322(TestPath, testimages, testlabels)
    # 将testimages和testlabels加入到trainimages和trainlabels中
    # trainimages, trainlabels, testimages, testlabels = AddTesttoTrain(trainimages, trainlabels, testimages, testlabels)
    # 将trainimages,trainlabels转换为numpy数组
    trainimages, trainlabels = load_dataset322(trainimages, trainlabels)
    testimages, testlabels = load_dataset322(testimages, testlabels)
    #对数据元进行处理
    trainx, trainy = preprocess322(trainimages, trainlabels)
    testx, testy = preprocess322(testimages, testlabels)
    #训练
    train322(trainx, trainy, testx, testy)


if __name__ == "__main__":
    main()