import tensorflow as tf
import numpy as np
from model import Model 
from para_model import D_model, G_model_3D, G_model_2D
from para_model import train_epoch
from para_data import target_class
from batch_fetch import next_batch
from test import attack_test
def train(is_3D = True):
    #*******************build session************
	#initialize & restore
	model = Model(is_3D)
	if is_3D:
		G_model = G_model_3D
	else:
		G_model = G_model_2D
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	sess = tf.InteractiveSession(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)
	print('[*] Loading discriminator from %s' %(D_model))
	model.saver_D.restore(sess, D_model)
	try:
		model.saver_G.restore(sess, G_model)
		print ('[*] Loading successful! Loading generator from %s ' % (G_model))
	except:
		print ('[*] No suitable checkpoint!')


	#********************train**************************

	for step in range(train_epoch):
		flag, clip_batch, label_batch, next_pos = next_batch(0, 'train')
		iteration = 0
		while flag:
			feed_dict = dict()
			feed_dict[model.x] = clip_batch
			feed_dict[model.c] = label_batch
			feed_dict[model.is_train] = True
			fetches = [model.p, model.loss, model.train]
			p_batch, loss_step, _ = sess.run(fetches,feed_dict)
			print("epoch %d iteration %d : %f" % (step+1, iteration+1, loss_step))
			if iteration != 0 and iteration%500 == 0:
				print("Model saved in file: %s" % model.saver_G.save(sess, G_model))
			if iteration != 0 and iteration%50 ==0:
			        res_target, res_non_target= attack_test(sess, p_batch[0], 'valid') ######
				print("the attack success rate for target inputs: %f, for non-target inputs %f"%
					(res_target, res_non_target))

			flag, clip_batch, label_batch, next_pos = next_batch(next_pos, 'train')
			iteration += 1

			

	sess.close()
	print('Training finised!')

if __name__ == '__main__':
	train(False)
