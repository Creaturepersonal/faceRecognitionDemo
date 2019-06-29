import tensorflow as tf
import numpy as np


# 加载模型
network = tf.keras.models.load_model('model/model.h5')
# 加载参数
network.load_weights('model/weights.ckpt')


#图片预测函数
def predict322(images):
    predict = network.predict(images)
    predict = tf.nn.softmax(predict)
    print(predict)
    if max(predict[0]) > 0.9:
        argmax = tf.argmax(predict, axis=-1)
        # result = np.array(argmax)[0]
        return argmax
    else:
        return None

#实时预测函数
def predictvideo322(images):
    # 预测
    predict = network.predict(images)
    predict = tf.nn.softmax(predict)
    # 找到概率最大的位置
    if max(predict[0]) > 0.70:
        argmax = tf.argmax(predict, axis=-1)
        result = np.array(argmax)[0]
        return result
    else:
        return None