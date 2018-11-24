# cite from C3D, modified some;
# reference: https://github.com/hx173149/C3D-tensorflow
import tensorflow as tf
from para_data import num_classes
from para_data import num_frames_per_clip, crop_size

assert num_frames_per_clip == 16
assert crop_size == 112

def discriminator(_X, _dropout=0.6, reuse=None):
	_weights = {
			'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00,reuse=reuse),
			'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00,reuse=reuse),
			'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00,reuse=reuse),
			'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00,reuse=reuse),
			'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00,reuse=reuse),
			'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00,reuse=reuse),
			'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00,reuse=reuse),
			'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00,reuse=reuse),
			'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001,reuse=reuse),
			'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002,reuse=reuse),
			'out': _variable_with_weight_decay('wout', [4096, num_classes], 0.04, 0.005,reuse=reuse)
			}
	_biases = {
			'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0,reuse=reuse),
			'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0,reuse=reuse),
			'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0,reuse=reuse),
			'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0,reuse=reuse),
			'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0,reuse=reuse),
			'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0,reuse=reuse),
			'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0,reuse=reuse),
			'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0,reuse=reuse),
			'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0,reuse=reuse),
			'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0,reuse=reuse),
			'out': _variable_with_weight_decay('bout', [num_classes], 0.04, 0.0,reuse=reuse),
			}
	d_vars = _weights.values()+_biases.values()
	saver_D = tf.train.Saver(d_vars)

	# Convolution Layer
	conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
	conv1 = tf.nn.relu(conv1, 'relu1')
	pool1 = max_pool('pool1', conv1, k=1)

	# Convolution Layer
	conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
	conv2 = tf.nn.relu(conv2, 'relu2')
	pool2 = max_pool('pool2', conv2, k=2)

	# Convolution Layer
	conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
	conv3 = tf.nn.relu(conv3, 'relu3a')
	conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
	conv3 = tf.nn.relu(conv3, 'relu3b')
	pool3 = max_pool('pool3', conv3, k=2)

	# Convolution Layer
	conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
	conv4 = tf.nn.relu(conv4, 'relu4a')
	conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
	conv4 = tf.nn.relu(conv4, 'relu4b')
	pool4 = max_pool('pool4', conv4, k=2)

	# Convolution Layer
	conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
	conv5 = tf.nn.relu(conv5, 'relu5a')
	conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
	conv5 = tf.nn.relu(conv5, 'relu5b')
	pool5 = max_pool('pool5', conv5, k=2)

	# Fully connected layer
	pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
	dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
	dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

	dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
	dense1 = tf.nn.dropout(dense1, _dropout)

	dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
	dense2 = tf.nn.dropout(dense2, _dropout)

	# Output: class prediction
	out = tf.matmul(dense2, _weights['out']) + _biases['out'] #
	out_softmax = tf.nn.softmax(out)

	return out_softmax, saver_D
 
# cite from C3D	
def conv3d(name, l_input, w, b):
	return tf.nn.bias_add(
		tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], 
			padding='SAME'),b)

# cite from C3D
def max_pool(name, l_input, k):
	return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], 
		strides=[1, k, 2, 2, 1], padding='SAME', name=name)

# cite from C3D
def _variable_on_cpu(name, shape, initializer, reuse):
	with tf.device('/cpu:0'):
		with tf.variable_scope('var_name',reuse=reuse) as var_scope:
			var = tf.get_variable(name, shape, initializer=initializer, trainable=False) # not trainable
	return var

# cite from C3D
def _variable_with_weight_decay(name, shape, stddev, wd, reuse):
	var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev),reuse=reuse)
	if wd is not None:
		weight_decay = tf.nn.l2_loss(var) * wd
		tf.add_to_collection('losses', weight_decay)
	return var
