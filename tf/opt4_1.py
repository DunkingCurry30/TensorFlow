#coding:utf-8
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))

y = tf.matmul(x,w1)

loss_mse = tf.reduce_mean(tf.square(y_-y))
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start+BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 500 == 0:
			print"After training %d steps, w1 is"%(i)
			print sess.run(w1)
	print"Final w1 is \n%s"%(sess.run(w1))
'''
GradientDescentOptimizer结果：
	Final w1 is 
	[[0.98019385]
 	 [1.0159807 ]]
	训练在20000轮后结果仍在优化

MomentumOptimizer结果:
	Final w1 is 
	[[1.0043069]
 	 [0.9948299]]
	训练在6500轮后结果不再变化

AdamOptimizer结果：
	Final w1 is 
	[[1.0043191]
   	 [0.9948099]]
	训练在7000轮后结果开始在小范围内波动
'''
