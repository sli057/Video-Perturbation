import tensorflow as tf 
import numpy as np 
from model import Generator 
from para_model import G_model_3D, G_model_2D
from para_model import G_path_3D, G_path_2D

def generate_p(is_3D=True):
	model = Generator(is_3D)
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
	model.saver_G.restore(sess, G_model)

	p_batch = model.p.eval(session =sess, feed_dict = {model.is_train: False})
	return p_batch

if __name__ == "__main__":
	p_batch = generate_p(False)
	print(np.array(p_batch).shape)
	# batch_size_per_power, num_frames_per_clip....
	np_name = G_path_2D + 'p_batch.npy'
	np.save(np_name, p_batch)
	print("The generated perturbations are saved in " + np_name)		
