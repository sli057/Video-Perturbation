
start_lr = 0.002
decay_steps = 2000
decay_rate = 0.95
train_epoch = 3 

batch_size_per_tower = 16 
gpu_number = 2
batch_size = batch_size_per_tower * gpu_number

z_size = 100
k_size = 3
p_max = 10
l = 1
ratio = 0.5

G_path_3D = "./G_model_3D/"
G_path_2D = "./G_model_2D/"
G_model_2D = G_path_2D + "G.ckpt"
G_model_3D = G_path_3D + "G.ckpt"
D_model = "./D_model/sports1m_finetuning_ucf101.model"


# descriminator : drop_out









