import tensorflow as tf
from para_model import z_size, k_size
from para_model import batch_size_per_tower as batch_size
from para_data import num_frames_per_clip, crop_size

assert num_frames_per_clip == 16
assert crop_size == 112

def generator_3D(input_z, is_train=True, reuse=False):
	z = tf.reshape(input_z,[batch_size,1,1,1,z_size])

	gen_dconv1 = deconv3d(z, [1,7,7,512,z_size], [batch_size,1,7,7,512],
		[1,1,1,1,1], 'VALID', reuse=reuse, name="gen_dconv1")
	gen_dconv1 = batch_norm(gen_dconv1, reuse, is_training=is_train, name="gen_nb_1")
	gen_dconv1 = tf.nn.relu(gen_dconv1)

	gen_dconv2 = deconv3d(gen_dconv1,[k_size,k_size,k_size,256,512], [batch_size,2,14,14,256],
		[1,2,2,2,1], 'SAME', reuse=reuse, name="gen_dconv2")
	gen_dconv2 = batch_norm(gen_dconv2, reuse, is_training=is_train, name="gen_nb_2")
	gen_dconv2 = tf.nn.relu(gen_dconv2)

	gen_dconv3 = deconv3d(gen_dconv2,[k_size,k_size,k_size,128,256], [batch_size,4,28,28,128],
		[1,2,2,2,1], 'SAME', reuse=reuse, name="gen_dconv3")
	gen_dconv3 = batch_norm(gen_dconv3, reuse=reuse, is_training=is_train, name="gen_nb_3")
	gen_dconv3 = tf.nn.relu(gen_dconv3)
	
	gen_dconv4 = deconv3d(gen_dconv3,[k_size,k_size,k_size,64,128], [batch_size,8,56,56,64],
		[1,2,2,2,1], 'SAME',reuse=reuse,  name="gen_dconv4")
	gen_dconv4 = batch_norm(gen_dconv4, reuse=reuse, is_training=is_train, name="gen_nb_4")
	gen_dconv4 = tf.nn.relu(gen_dconv4)
	
	gen_dconv5 = deconv3d(gen_dconv4,[k_size,k_size,k_size,3,64], [batch_size,16,112,112,3],
		[1,2,2,2,1], 'SAME', reuse=reuse, name="gen_dconv5")
	out = tf.nn.tanh(gen_dconv5)
	return out


def generator_2D(input_z, is_train=True, reuse=False):
	z = tf.reshape(input_z,[batch_size,1,1,1,z_size])

	gen_dconv1 = deconv3d(z, [1,7,7,512,z_size], [batch_size,1,7,7,512],
		[1,1,1,1,1], 'VALID', reuse=reuse, name="gen_dconv1")
	gen_dconv1 = batch_norm(gen_dconv1, reuse, is_training=is_train, name="gen_nb_1")
	gen_dconv1 = tf.nn.relu(gen_dconv1)

	gen_dconv2 = deconv3d(gen_dconv1,[k_size,k_size,k_size,256,512], [batch_size,1,14,14,256],
		[1,1,2,2,1], 'SAME', reuse=reuse, name="gen_dconv2")
	gen_dconv2 = batch_norm(gen_dconv2, reuse, is_training=is_train, name="gen_nb_2")
	gen_dconv2 = tf.nn.relu(gen_dconv2)

	gen_dconv3 = deconv3d(gen_dconv2,[k_size,k_size,k_size,128,256], [batch_size,1,28,28,128],
		[1,1,2,2,1], 'SAME', reuse=reuse, name="gen_dconv3")
	gen_dconv3 = batch_norm(gen_dconv3, reuse=reuse, is_training=is_train, name="gen_nb_3")
	gen_dconv3 = tf.nn.relu(gen_dconv3)
	
	gen_dconv4 = deconv3d(gen_dconv3,[k_size,k_size,k_size,64,128], [batch_size,1,56,56,64],
		[1,1,2,2,1], 'SAME',reuse=reuse,  name="gen_dconv4")
	gen_dconv4 = batch_norm(gen_dconv4, reuse=reuse, is_training=is_train, name="gen_nb_4")
	gen_dconv4 = tf.nn.relu(gen_dconv4)
	
	gen_dconv5 = deconv3d(gen_dconv4,[k_size,k_size,k_size,3,64], [batch_size,1,112,112,3],
		[1,1,2,2,1], 'SAME', reuse=reuse, name="gen_dconv5")
	out = tf.nn.tanh(gen_dconv5)
	out_tile = tf.tile(out,[1,16,1,1,1])

	return out_tile


# some helper functions for define layers
def deconv3d(input, filter_shape, output_shape,
	strides, padding,reuse, stddev=0.02, 
	name="deconv3d"):
	#input shape: [batch_size, frame, height, width, in_channels]
	#filter_shape: [depth, height, width, output_channels, in_channels]
	#output_shape: [batch_size, frame, height, width, output_channels]
	#strides: for each input dimension
	with tf.variable_scope(name,reuse=reuse):
		w = tf.get_variable('w',filter_shape,
			initializer=tf.random_normal_initializer(stddev=stddev))
		deconv = tf.nn.conv3d_transpose(input, w, output_shape=output_shape,
			strides=strides, padding=padding)
		bias = tf.get_variable('bias',[output_shape[-1]],
			initializer=tf.constant_initializer(0.0))
		deconv = tf.nn.bias_add(deconv,bias);
	return deconv

def batch_norm(input, reuse, is_training, name="batch_norm"):
	return tf.contrib.layers.batch_norm(input,
		decay=0.1,
		updates_collections=None,
		epsilon=1e-5,
		scale=True,
		is_training=is_training,
		reuse=reuse,
		scope=name)