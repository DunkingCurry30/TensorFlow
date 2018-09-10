#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

tfRecord_train = './data/mnist_train.tfrecords'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'

mnist = input_data.read_data_sets('./data/',one_hot=True)

image_train = mnist.train.images
label_train = mnist.train.labels
image_test = mnist.test.images
label_test = mnist.test.labels
train_num = mnist.train.num_examples
test_num = mnist.test.num_examples

def write_tfRecord(tfRecordName,num,images,labels):
	writer = tf.python_io.TFRecordWriter(tfRecordName)
	num_pic = 0

	for i in range(num):
		img_raw = images[i].tobytes()
		label_raw = labels[i].astype(np.int64)
		example = tf.train.Example(features=tf.train.Features(feature={
			'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
			'label':tf.train.Feature(int64_list=tf.train.Int64List(value=label_raw))
			}))
		writer.write(example.SerializeToString())
		num_pic += 1
		print('the number of pictures:',num_pic)
	writer.close()
	print('write tfrecord successful')

def generate_tfRecord():
	isExists = os.path.exists(data_path)
	if not isExists:
		op.makedirs(data_path)
		print("The directionary was created successfully.")
	else:
		print("Directionary already exists")
	write_tfRecord(tfRecord_train,train_num,image_train,label_train)
	write_tfRecord(tfRecord_test,test_num,image_test,label_test)

def read_tfRecord(tfRecord_path):
	filename_queue = tf.train.string_input_producer([tfRecord_path])
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
										features={
											'label':tf.FixedLenFeature([10],tf.int64),
											'img_raw':tf.FixedLenFeature([],tf.string)
										})
	img = tf.decode_raw(features['img_raw'],tf.uint8)
	
	img.set_shape([784])
	img = tf.cast(img,tf.float32)*(1./255)
	label = tf.cast(features['label'],tf.uint8)
	return img,label

def get_tfrecord(num,isTrain=True):
	if isTrain:
		tfRecord_path = tfRecord_train
	else:
		tfRecord_path = tfRecord_test
	img,label = read_tfRecord(tfRecord_path)
	#实现随机读取一个batch的数据
	img_batch,label_batch = tf.train.shuffle_batch([img,label],batch_size=num,num_threads =2,
													capacity = 1000,#队内元素的最大数量
													min_after_dequeue = 700)#出队后队内元素的最小数量，用于确保元素的混合级别
	return img_batch,label_batch

def main():
	generate_tfRecord()

if __name__=='__main__':
	main()






















