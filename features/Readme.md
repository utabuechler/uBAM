Here comes the description of this part.

pytorch: version 0.4.0, build with cuda 9.1
tensorflow: version 1.8.0, build with cuda 9.1

for tensorboard:
- run
CUDA_VISIBLE_DEVICES=1 tensorboard --logdir='/export/home/ubuechle/Mouse_Analysis/NatureMethods_code/input_output/rats/features/pytorch/checkpoints/logs/' --port=6006
on the server in tf environment
- run
ssh -N -f -L 129.206.117.63:8008:localhost:6006 ubuechle@compgpu1
on own pc

