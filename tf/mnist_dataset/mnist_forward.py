#coding:utf-8
import tensorflow as tf

INPUT_NODE = 784#每张图片28x28共784个像素点，即784个输入节点
OUTPUT_NODE = 10#输出10个数，每个数代表索引号对应数字的概率
LAYER1_NODE = 500#隐藏层的节点个数

def get_weight(shape,regularizer):
	#tf.truncated_normal相比与random_normal,会去掉偏离较大的正态分布点
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: 
		#使用l2正则化
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def forward(x,regularizer):
	w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
	
	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1,w2)+b2
	
	return y
