#test the attack rate given perturbation
import tensorflow as tf
import numpy as np
from model import Discriminator
from para_model import D_model
from para_data import target_class, crop_mean
from batch_fetch import next_batch

model = Discriminator()

def attack_test(sess, p, mode='test'):
	model.saver_D.restore(sess, D_model)
	p_batch = np.expand_dims(p,axis=0) #p_batch
	res_non_target=[]
	res_target=[]
	if mode == "valid":
		filename = '../20BN_JESTER/our_val.list'
	elif mode == "test":
		filename = '../20BN_JESTER/our_test.list'
	flag, clip_batch, label_batch, next_pos = next_batch(0, mode)
	
	while flag:
		test_batch = clip_batch+p
		test_batch = np.clip(test_batch+crop_mean,0,255) - crop_mean
		predict_score = model.softmax_score.eval(
			session=sess,
			feed_dict = {model.x_x:test_batch})

		top1_predicted_label = np.argmax(predict_score, axis=1)
		true_label = np.array(label_batch)
		
		target_exist = true_label==target_class
		cmp_label = (true_label==top1_predicted_label).astype(float)
		num_target = sum(0+target_exist)
		num_non_target = sum(1-target_exist)
		if num_target!=0:
			res_target.append(sum(cmp_label[target_exist])/num_target)
		if num_non_target!=0:
			res_non_target.append(sum(cmp_label[np.logical_not(target_exist)])/num_non_target)
		flag, clip_batch, label_batch, next_pos = next_batch(next_pos, mode)
		print(next_pos)
	res_target = 1 - np.mean(res_target)
	res_non_target = np.mean(res_non_target)
	return res_target, res_non_target ###loss

if __name__ == '__main__':
	p_path = "./G_model_2D/p_batch.npy"
	test_idx = 0 #test the ith perturbation
	test_shift = 0 #shift the perturbation
	gen_sample = np.load(p_path)
	# gen_sample = np.zeros(gen_sample.shape) #baseline, no perturbation
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	sess = tf.InteractiveSession(config=config)
	gen_sample = gen_sample[test_idx]
	gen_sample = np.roll(gen_sample, test_shift)
	res_target, res_non_target = attack_test(sess, gen_sample, mode="test")
	
	print("the attack success rate for target inputs: %f"%res_target)
	print("the attack success rate for non-target inputs: %f"%res_non_target)

