import os
home = os.path.expanduser("..")

_project=project='rats'
_input_path = home+'/resources/'

results_path = home+'/output/'+_project+'/'
_output_path = home+'/output/'+_project+'/'

if not os.path.exists(_output_path):
    os.makedirs(_output_path)

###########################
# data configurations
###########################
crops_path  = _input_path+_project+'/crops/'
frames_path = _input_path+_project+'/data/'
detection_file = _input_path+_project+'/OptoReha_cohortsdays_UTF16.csv'

video_path = _input_path+_project+'/data/'
video_format = 'MTS'
frames_crop = _input_path+_project+'/vids_cropping.cvs'
seq_len = 8

###########################
# network configurations
###########################
###default values, no need to change
net_name = 'lstm_walking'
features_path = _input_path+_project+'/features/'

init_weights = _input_path+_project+'/features/pytorch/"
checkpoint_path = _input_path+_project+'/features/'
final_weights = checkpoint_path+'lstm_checkpoint.pth.tar'

input_img_size = 227
seq_len = 8
gpu_id = 0

batchsize_train=48
batchsize_test = 6
lr = 0.01#base learning rate
input_img_size = 227#width and size of the images as input to the network
optimizer='SGD'#SGD or Adam
skip_frames=0

augment = True

train_display_freq = 10#frequency of outputs during training
train_nr_epochs = 50#number of epochs
train_checkpoint_freq = 1000#frequency of saving snapshots
train_stepsize = 25000#frequency of changing the learning rate

##############################
# For Generative Model (VAE)
##############################
vae_weights_path = _input_path+_project+'/magnification/'
encode_dim = 50

vae_checkpoint_path = _output_path+_project+'/magnification/training/'

# Network
vae_encode_dim = 50
feature_size = 4096

#Training
vae_n_epochs = 100
vae_lr = 5e-4
vae_bs = 35
vae_verbose_idx = 20
vae_weight_decay = 0.00001
vae_step_size = 20
vae_gamma2 = 0.3
vae_kernels = 8
vae_alpha = 1
vae_beta = 0.001
vae_gamma = 0.001
vae_delta = 1
vae_size = 60000
vae_savename = 'interpolation'

##############################
# For the text
##############################
RED   = '\033[91m'
END   = '\033[0m'
