

import tensorflow as tf

class fc(tf.Module):
	def __init__(self, output_dim):
		self.output_dim = output_dim
		self.b = tf.Variable(tf.constant(0.0, shape=[self.output_dim]))
	def __call__(self, x):
		if not hasattr(self, 'w'):
			w_init = tf.random.truncated_normal([x.shape[1], self.output_dim], stddev=0.1)
			self.w = tf.Variable(w_init)

		return tf.matmul(x, self.w) + self.b

class conv(tf.Module):
	def __init__(self, output_dim, filterSize, stride):
		self.filterSize = filterSize
		self.output_dim = output_dim
		self.stride = stride
		self.b = tf.Variable(tf.constant(0.0, shape=[self.output_dim]))
	def __call__(self, x):
		if not hasattr(self, 'w'):
			w_init = tf.random.truncated_normal([self.filterSize, self.filterSize, x.shape[3], self.output_dim], stddev=0.1)
			self.w = tf.Variable(w_init)
		x = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding='SAME') + self.b
		return tf.nn.relu(x)

class maxpool(tf.Module):
	def __init__(self, poolSize):
		self.poolSize = poolSize

	def __call__(self, x):
		return tf.nn.max_pool2d(x, ksize=(1, self.poolSize, self.poolSize, 1),
								strides=(1, self.poolSize, self.poolSize, 1), padding='SAME')

class flat(tf.Module):
	def __call__(self, x):
		inDimH = x.shape[1]
		inDimW = x.shape[2]
		inDimD = x.shape[3]
		return tf.reshape(x, [-1, inDimH * inDimW * inDimD])

class unflat(tf.Module):
	def __init__(self, outDimH, outDimW, outDimD):
		self.new_shape = [-1, outDimH, outDimW, outDimD]

	def __call__(self, x):
		x = tf.reshape(x, self.new_shape)
		return x