#coding:utf-8
import tensorflow as tf 
import numpy as np 
BATCH_SIZE = 8
SEED = 23455
COST = 9
PROFIT = 1

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))

y = tf.matmul(x,w1)

#自定义损失函数
loss =tf.reduce_sum(tf.where(tf.greater(y,y_),COST*(y-y_),PROFIT*(y_-y))) 
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
		if i % 500 == 0:
			print "After %d trainings w1 is "%(i)
			print sess.run(w1)
	print"Final w1 is"
	print sess.run(w1)
'''
此例中，成本远高于利润，使用自定义损失函数，模型会尽量往少了预测
结果也验证了相较于均方误差的预测偏小
'''
