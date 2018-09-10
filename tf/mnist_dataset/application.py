#coding:utf-8
#使用训练好的全连接神经网络预测手写数字图片
import cv2
import numpy as np
import tensorflow as tf
import forward
import backward

def restore_model(test_pic_arr):
	#实例化tf.Graph类，定义整个计算的数据流图，默认有一张（单个流程可以不写），存在多个线程必须添加这句
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32,(None,forward.INPUT_NODE))
		y = forward.forward(x,None)
		#前向传播预测的10个概率值取最大
		preValue = tf.argmax(y,1)
		
		#滑动平均模型
		variables_average = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variables_average.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
	
		with tf.Session() as sess:
			#加载模型进行预测
			ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				
				preValue = sess.run(preValue,feed_dict={x:test_pic_arr})
				return preValue
			else:
				print('No checkpoint found!')
				return -1


def pre_pic(test_img):
	img = cv2.imread(test_img,0)
	img2 = cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
	#二值化
	threshold = 50
	for i in range(img2.shape[0]):
		for j in range(img2.shape[1]):
			img2[i][j] = 255-img2[i][j]
			if img2[i][j] < threshold:
				img2[i][j] == 0
			else:
				img2[i][j] == 1
	#构建成1维数组，且数据类型为float32
	img2 = img2.reshape([1,784]).astype(np.float32)
	return img2

def application():
	testNum = input('input the number of test pictures:')
	for i in range(testNum):
		test_pic =raw_input("the path of test picture:")
		#对输入图片进行预处理，使之符合神经网络输入
		test_pic_arr = pre_pic(test_pic)
		#喂入神经网络，进行预测
		preValue = restore_model(test_pic_arr)
		#打印预测值
		print('the prediction of the test picture is %s'%preValue)

def main():
	application()

if __name__=='__main__':
	main()
