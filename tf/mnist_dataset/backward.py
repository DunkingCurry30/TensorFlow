#coding:utf-8

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import forward
import os
import time

BATCH_SIZE = 100#较200提高了50%左右
LEARNING_RATE_BASE = 0.8#学习率基数
LEARNING_RATE_DECAY = 0.99#学习率衰减率
STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99#滑动平均衰减率
REGULARIZER = 0.0001#参数w在总loss中的比重
MODEL_SAVE_PATH = './model1/'#模型保存路径
MODEL_NAME = 'mnist_model'#模型保存名称

def backward(mnist):
	x = tf.placeholder(tf.float32,[None,forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
	y = forward.forward(x,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)
	
	#交叉熵
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	
	#正则化损失函数
	loss = cem + tf.add_n(tf.get_collection('losses'))
	
	#指数衰减学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)
	
	#定义反向传播方法，梯度下降效果最好
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	

	#滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
	ema_op = ema.apply(tf.trainable_variables())
	
	#在训练神经网络时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
	#又要更新每个参数的滑动平均值，为了一次完成多个操作，
	#使用tf.control_dependencies([a,b]),使a,b具有依赖关系
	with tf.control_dependencies([train_step,ema_op]):
		#tf.no_op不进行任何操作
		train_op = tf.no_op(name='train')
	#声明tf.train.Saver类用于保存模型
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		#断点续训，在上次训练的模型基础上继续训练
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)
		
		for i in range(STEPS):
			#mnist.train.next_batch，随机取一小部分数据
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			#
			_,loss_val,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
			if i % 1000 == 0:
				print('After %d training step(s), loss on training batch is %s'%(step,loss_val))
				#每个1000轮的模型保存到制定目录
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
		
def main():
	start = time.clock()
	#读取数据集
	mnist = input_data.read_data_sets('./data/',one_hot=True)
	backward(mnist)
	end = time.clock()
	print end-start

if __name__=='__main__':
	main()
