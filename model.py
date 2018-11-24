import tensorflow as tf
from para_model import batch_size_per_tower, gpu_number, batch_size
from para_model import z_size, p_max, l, ratio 
from para_model import start_lr, decay_steps, decay_rate
from para_data import num_frames_per_clip, crop_size, num_channels
from para_data import num_classes
from discriminator import discriminator
from generator import generator_2D, generator_3D

assert gpu_number*ratio == int(gpu_number*ratio)



class Model:
	def __init__(self, is_3D=True):
		self.x = tf.placeholder(tf.float32, [batch_size, num_frames_per_clip,
				crop_size, crop_size, num_channels], name='clip')
		self.c = tf.placeholder(tf.int32, [batch_size], name='true_label')
		self.is_train = tf.placeholder(tf.bool, [], name='batch_norm_is_train')
		global_step = tf.Variable(0,trainable=False)
		lr = tf.train.exponential_decay(start_lr, global_step,
				decay_steps=decay_steps, decay_rate=decay_rate,staircase=True)
		g_optim = tf.train.AdamOptimizer(lr,beta1=0.3)

		tower_grads=[]
		tower_losses= []
		
		for gpu_index in range(gpu_number):
			with tf.device('/gpu:%d'%gpu_index):
				x_tower = self.x[gpu_index*batch_size_per_tower:(gpu_index+1)*batch_size_per_tower,:,:,:,:]
				c_tower = self.c[gpu_index*batch_size_per_tower:(gpu_index+1)*batch_size_per_tower]
				c_one_hot = tf.one_hot(c_tower,num_classes)
				z = tf.random_normal([batch_size_per_tower, z_size])
				shift = tf.random_uniform([], minval=0, maxval = num_frames_per_clip, dtype = tf.int32, name = "shift")
				if is_3D:
					z_out = generator_3D(z, self.is_train, reuse=tf.AUTO_REUSE)
					z_out = tf.manip.roll(z_out, shift=shift, axis=1)
				else:
					z_out = generator_2D(z, self.is_train, reuse=tf.AUTO_REUSE)
			
				p = tf.scalar_mul(p_max, z_out)
				d_softmax, saver_D = discriminator(x_tower+p, reuse=tf.AUTO_REUSE)

				if gpu_index < gpu_number*ratio:
					loss = l*tf.reduce_mean(tf.reduce_sum(-c_one_hot*tf.log(
						tf.clip_by_value(1-d_softmax,1e-10,1.0)),axis=1))
				else:
					loss = tf.reduce_mean(tf.reduce_sum(-c_one_hot*tf.log(
						tf.clip_by_value(d_softmax,1e-10,1.0)),axis=1))
			
				g_vars = [var for var in tf.trainable_variables() if 'gen_'in var.name]
				grads = g_optim.compute_gradients(loss, var_list=g_vars)
				tower_grads.append(grads)
				tower_losses.append(loss)
		
		loss_ave = tf.reduce_mean(tower_losses)
		grads_ave = self.average_gradients(tower_grads)
		train_G = g_optim.apply_gradients(grads_ave,global_step=global_step)
		saver_G = tf.train.Saver(g_vars)
		self.p = p
		self.saver_D = saver_D
		self.saver_G = saver_G
		self.loss = loss_ave
		self.train = train_G

	def average_gradients(self, tower_grads):
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			grads = []
			for g, _ in grad_and_vars:
				expanded_g = tf.expand_dims(g, 0)
				grads.append(expanded_g)
			grad = tf.concat(grads, 0)
			grad = tf.reduce_mean(grad, 0)
			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
		return average_grads

class Generator:
	def __init__(self, is_3D=True):
		self.is_train = tf.placeholder(tf.bool, [], name='batch_norm_is_train')

		with tf.device('/gpu:%d'%0):
			z = tf.random_normal([batch_size_per_tower, z_size])
			if is_3D:
				z_out = generator_3D(z, self.is_train, reuse=tf.AUTO_REUSE)
			else:
				z_out = generator_2D(z, self.is_train, reuse=tf.AUTO_REUSE)
			p = tf.scalar_mul(p_max, z_out)
		self.p = p 
		self.saver_G = tf.train.Saver(tf.trainable_variables())


class Discriminator:
	def __init__(self):
		self.x_x = tf.placeholder(tf.float32, [batch_size, num_frames_per_clip,
			crop_size, crop_size, num_channels])
		res_softmax_score = []
		for gpu_index in range(gpu_number):
			with tf.device('/gpu:%d'%gpu_index):
				x_x_tower = self.x_x[gpu_index*batch_size_per_tower:(gpu_index+1)*batch_size_per_tower, :, :, :, :]
				if gpu_index == 0:
					softmax_score, saver_D = discriminator(x_x_tower, reuse=False)
				else:
					softmax_score, _ = discriminator(x_x_tower,reuse=True)
				res_softmax_score.append(softmax_score)
		self.saver_D = saver_D
		self.softmax_score = tf.concat(res_softmax_score, 0)


