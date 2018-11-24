# this is to fetch batch from tran_class_* valid_class* test_class* directories
import numpy as np
from para_model import ratio
from para_model import batch_size
from para_data import permutation_train_target_class
from para_data import permutation_train_non_target_class
from para_data import pre_sum_train, pre_sum_valid, pre_sum_test
from para_data import num_classes, np_path
def num_to_npy(idx, pre_sum, dir_name):
	label = 0
	while (label+1<num_classes and pre_sum[label+1]<=idx ):
		label+=1
	file_cnt = idx - pre_sum[label]
	clip_name = '/'.join([np_path, dir_name+str(label).zfill(3), 
		str(file_cnt).zfill(3)+".npy"])	
	return label, np.load(clip_name)


def next_batch(pos, mode="train"):
	if mode == "train":
		pre_sum = pre_sum_train
		dir_name = "train_class_"
	elif mode == "valid":
		pre_sum = pre_sum_valid
		dir_name = "valid_class_"
 	elif mode == "test":
		pre_sum = pre_sum_test
		dir_name = "test_class_"
	else:
		raise ValueError("mode must be train, valid, or test")

	nex_pos = pos + batch_size
	ret_label = []
	ret_clip = []

	if mode == "train":
		total = len(permutation_train_non_target_class)
	else:
		total = pre_sum[num_classes]

	if nex_pos > total:
		return False, ret_clip, ret_label, nex_pos

	for i in range(pos, nex_pos):
		if mode=="train":
			if i < pos+batch_size*ratio:
				idx=permutation_train_target_class[i%len(permutation_train_target_class)]
			else:
				idx=permutation_train_non_target_class[i]
		else:
			idx=i;
		label, clip = num_to_npy(idx, pre_sum, dir_name)
		ret_label.append(label)
		ret_clip.append(clip)
	return True, ret_clip, ret_label, nex_pos

