import numpy as np

target_class = 1
num_classes = 101

crop_size = 112
num_frames_per_clip = 16
num_channels = 3

crop_mean = np.load("./data/crop_mean.npy")
np_path = "./data"

cnt_class_train = np.load('/'.join([np_path,"cnt_class_train.npy"]))
cnt_class_valid = np.load('/'.join([np_path,"cnt_class_valid.npy"]))
cnt_class_test = np.load('/'.join([np_path,"cnt_class_test.npy"]))


#cnt_class=[168,134,179]
assert len(cnt_class_train) == num_classes
assert len(cnt_class_valid) == num_classes
assert len(cnt_class_test) == num_classes
pre_sum_train = [0]*(num_classes+1)
pre_sum_valid = [0]*(num_classes+1)
pre_sum_test = [0]*(num_classes+1)
for i in range(num_classes):
	pre_sum_train[i+1] = pre_sum_train[i] + cnt_class_train[i]
	pre_sum_valid[i+1] = pre_sum_valid[i] + cnt_class_valid[i]
	pre_sum_test[i+1] = pre_sum_test[i] + cnt_class_test[i]

permutation_train_target_class = np.random.permutation(range(pre_sum_train[target_class],pre_sum_train[target_class+1]))
permutation_train_non_target_class = list(set(range(pre_sum_train[num_classes]))-set(permutation_train_target_class))			

print("The total number of training clip is %d. "%pre_sum_train[num_classes])
print("The total number of valid clip is %d. "%pre_sum_valid[num_classes])
print("The total number of test clip is %d."%pre_sum_test[num_classes])
