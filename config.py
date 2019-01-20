
_input_path  = '/export/scratch/ubuechle/Mouse_Analysis/NatureMethods_code/input_output/'
_output_path = '/export/scratch/bbrattol/behavior_analysis_demo/'
_project='rats'


###########################
# Input data configurations
###########################
video_path = '/export/home/ubuechle/Mouse_Analysis/videos/OptoReha/'
video_format = 'MTS'
detection_file = _input_path+_project+'/preprocessing/detections/OptoReha_cohortsdays.csv'

###########################
# Pre-processiong configurations
###########################
frames_path = '/export/home/ubuechle/Mouse_Analysis/OptoReha/data/'
frames_crop = _input_path+_project+'/preprocessing/vids_cropping.cvs'
crops_path  = _input_path+_project+'/preprocessing/crops/'

##############################
# Path saving for results
##############################
results_path = _output_path+_project



###########################
# network configurations
###########################
net_name = 'lstm_'+_project     #only shown in the prototxt file

features_path = _input_path+_project+'/features/pytorch/'
init_weights = _input_path+_project+'/features/pytorch/checkpoints_twoLosses_fromImagenet_beta001/checkp_epoch_013_iter_26000.pth.tar'#in case we want to initialize with weights, if no weights are used as initialization: init_weights = None

checkpoint_path = results_path+'/features/pytorch/checkpoints_twoLosses_fromImagenet_beta001/'
final_weights = checkpoint_path+'checkp_epoch_025_iter_50000.pth.tar'#None#fill in after the network is trained, if still empty when extracting the features the last output of the trained network will be used

seq_len = 8

###default values, no need to change
#!!!!!!consider: If you lower the batchsize also lower the learning rate!!!!!!
batchsize_train=48
batchsize_test = 6
lr = 0.01           #base learning rate
input_img_size = 227#width and size of the images as input to the network
optimizer='SGD'     #SGD or Adam
skip_frames=0

augment = True

#parameters for solver
train_display_freq = 10     #frequency of outputs during training
train_nr_epochs = 50        #number of epochs
train_checkpoint_freq = 1000#frequency of saving snapshots
train_stepsize = 25000      #frequency of changing the learning rate

##############################
# For Case Study
##############################
# Behavior Comparison
min_train = 2000

##############################
# For Generative Model (VAE)
##############################
vae_checkpoint_path = results_path+'/magnification/training/'
model_name = 'interpolation_whole_dataset'
encode_dim = 20

vae_weights_path = vae_checkpoint_path+'VAE-5_Date-2019-1-15-17-28_interpolation_whole_dataset/'

##############################
# For the text
##############################

RED   = '\033[91m'
END   = '\033[0m'
