#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import mnist_forward
import mnist_generateds

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 7500
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'
train_num_examples = 55000

def backward(mnist):
	x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
	y = mnist_forward.forward(x,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)
	
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))
	
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		train_num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase = True)

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name='train')
	
	saver = tf.train.Saver()
	img_batch,label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE,isTrain=True)#获取随机样本
	print(type(img_batch),type(label_batch))
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)
		
		#多线程数据处理框架
		#开启线程协调器，实例化Coordinator类来协同启动多线程
		coord = tf.train.Coordinator()
		#明确调用tf.train.start_queue_runners来启动所有线程
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)

		for i in range(STEPS):
			#xs,ys = mnist.train.next_batch(BATCH_SIZE)
			xs,ys = sess.run([img_batch,label_batch])#现在需要用自己写的get_tfrecord函数来获取样本
			
			_,loss_val,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:xs})
			if i % 2500 == 0:
				print('After %d training step(s), loss on training batch is %s'%(step,loss_val))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
			
			#使用tf.train.Coordinator来停止所有线程
		coord.request_stop()
		coord.join(threads)

def main():
	mnist = input_data.read_data_sets('./data/',one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()
