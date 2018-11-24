#transpose UCF101 image file to npy file
import os
import numpy as np
import PIL.Image as Image
import cv2

UCF_path = "../"
num_class = 101
num_frames_per_clip = 16
num_channel = 3
crop_size = 112
mean_file = "crop_mean.npy"
np_mean = np.load(mean_file).reshape([num_frames_per_clip, crop_size, crop_size, num_channel])


def data_processing(mode):
	list_file = mode + ".list"
	dir_name_prefix = mode + "_class_"
	cnt_name = "cnt_class_" + mode
	stride_step = 1

	cnt_class=[0]*num_class

	lines = list(open(list_file, 'r'))
	for line in lines:
		line = line.strip('\n').split()
		image_path = line[0]
		label = int(line[1])
		dir_name = dir_name_prefix + str(label).zfill(3)
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		seg_dir = UCF_path + image_path
		print("I am extracting data from %s " % seg_dir)
		images = sorted(get_subfiles(seg_dir))
		num_images = len(images)
		if num_images < num_frames_per_clip:
			continue
		s_index = 0
		while (s_index+num_frames_per_clip <= num_images):
			one_clip = []
			for i in range(s_index, s_index+num_frames_per_clip):
				image_path = '/'.join([seg_dir,images[i]])
				img = np.array(Image.open(image_path))
				one_clip.append(img)
			one_clip = pre_process(one_clip)
			clip_name = '/'.join([dir_name,str(cnt_class[label]).zfill(3)])
			np.save(clip_name, one_clip)
			cnt_class[label] += 1
			s_index += stride_step
	print(cnt_class)
	np.save(cnt_name,cnt_class)	



def get_subdirs(dir):
    #"Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]
def get_subfiles(dir):
    #"Get a list of immediate subfiles"
    return next(os.walk(dir))[2]

def pre_process(clip_np):
	ret_clip=[]
	for i in range(num_frames_per_clip):
		img = Image.fromarray(clip_np[i].astype(np.uint8))
		if img.width > img.height :
        		scale = float(crop_size) / float(img.height)
        		img = np.array(cv2.resize(np.array(img),(int(img.width*scale+1), crop_size))).astype(np.float32)
        	else:
          		scale = float(crop_size)/float(img.width)
          		img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height*scale+1)))).astype(np.float32)
        	img = img[int((img.shape[0]-crop_size)/2):int((img.shape[0]-crop_size)/2 + crop_size), int((img.shape[1]- 		
			crop_size)/2):int((img.shape[1]-crop_size)/2 + crop_size),:] - np_mean[i]
		ret_clip.append(img)
	ret_clip = np.array(ret_clip)
	#print(ret_clip.shape)
	assert ret_clip.shape == (num_frames_per_clip, crop_size, crop_size, num_channel)
	return ret_clip

if __name__ == '__main__':
	mode = "train"
	data_processing(mode)




			







