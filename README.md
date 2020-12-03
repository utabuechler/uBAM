Here we collect all our code for our nature journal.

#System requirements
The software was tested on `Ubuntu 18.04`.
The code runs on Python 3. We suggest to install the newest version of Anaconda.

##Dependencies
You can install all required dependencies using the following command.
```
pip install -r requirements.txt
```

### CPU-only dependencies
The previous command automatically installs the gpu version of pytorch which is necessary for training the model. The demo does not need to run on a GPU, since it uses a pretrained model. Therefore you can install the cpu version of pytorch running the following command
```
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### GPU dependencies
Pytorch requires CUDA to be installed to run the network on the GPU. The cuda version required depends on the Ubuntu system, the GPU type and the pytorch version. We used CUDA 9.2 for our experiments.

##Hardware requirements
### Training and feature extraction
Type: Workstation
Processor: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
RAM: DDR2 512GB
Memory: standard HardDisk
GPU: single TITAN Xp 12GB

### Demo
Type: Laptop
Processor: Intel(R) Core(TM) i7-3537U CPU @ 2.00GHz
RAM: DDR2 8GB
Memory: SSD
GPU: None


#Installation guide
Python code does not need to be compiled, therefore the installation process is simply downloading the code and the sample data.
**During this phase we only provide a small sample of data for testing the demo locally. Training a functional model is not possible with this data.**

## Download data samples
Dowload the zip file at the following link
[https://heibox.uni-heidelberg.de/d/c67db8fa39474c13a5c9/](https://heibox.uni-heidelberg.de/d/c67db8fa39474c13a5c9/)
and unzip the content into the *resource/* folder. The folder should have the following structure:

- behaviorAnalysis/
- bin/
- resources/
  - README.md
  - humans/
    - data/
    - features/
    - magnification/
    - detections.csv
  - rats/
    - data/
    - features/
    - magnification/
    - detections.csv


##Installation time
Since the software is developed in python, there is no need to compile. Therefore the installation time is the same needed to install the dependencies and download the data. This should not take more than one hour, but it strongly depends on the system status and internet speed.

#Demo
The demo includes an interactive interface that utilizes the pre-trained model for nearest neighbor search, 2D projection and magnification.
The demo can be accessed in three way: remote server, using this code or docker image.

## Locally
Everything needed to run the interface locally on your machine is available in this folder. After installing all dependencies and downloading the sample data (as described in the previous sections above) run the following commands:
```
./bin/human_interface.sh
```
or the rats dataset
```
./bin/human_interface.sh
```

Note: the sample data used for the demo includes only two subjects for healthy and two impaired, which is not enough for training the model.

## Docker image
We provide a docker image which includes the sample data, the code and all dependencies are installed. Make sure that docker is installed on your system.
To use the docker image, first load the image in your local docker and start the container using
```
xhost +local:docker
docker load -i ubam.tar
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -i --name ubam -t ubam /bin/bash
```
The -v and -e options make sure that the interface is forwarded from the container to the host pc. This has been tested on ubuntu and it might change on MAC system.
When attached to the docker container, run the following to start the interface
```
cd /home/ubam/interface
./bin/human_interface.sh
```
Note: the sample data used for the demo includes only two subjects for healthy and two impaired, which is not enough for training the model.

##How to use
1. Select the video from the topdown menu in the top left side of the interface and click _Load Video_
2. Select the modality: postures or sequences.
3. Select the **query** using the buttons at the bottom of the preview
4. Choose the analysis:
  1. Nearest neighbor: show postures/sequences similar to the query.
  2. Low-dimensional 2D embedding (e.g. tSNE). Embed a subset of the data into a 2D embedding.
  3. Magnification. Amplify the deviation between the query and the nearest healthy reference.
5. Click the "show" button.

##Output
1. Nearest neighbor: show postures/sequences similar to the query.
2. Low-dimensional 2D embedding (e.g. tSNE). Embed a subset of the data into a 2D embedding.
3. Magnification. Amplify the deviation between the query and the nearest healthy reference.

##Runtime
Based on the type of experiment, the modality and internet speed (if remote), each click of the "show" button can take from 10 seconds up to 1 minute to show the result.

#Run on your data
To train a model on your data, create a configuration file in _behaviorAnalysis/_ for example by copying _config_pytorch_rats.py_ and changing the name to _config_pytorch_{yourdataset}.py_. Then change the parameters into the new config file, most importantly you have to change the following:
- project: the dataset name
- video-path: the root-path to your videos. The code will recursively looking for videos within each folder that matches the format in video-format and produce a csv file with all videos.
- video-format: the format of the videos, for example "avi", "MTS" or "mp4".

Our preprocessing code will extract each frame from all videos and save it as jpg, moreover it produces the necessary files and folders for the next steps.
```
cd behaviorAnalysis/
cp config_pytorch_{yourdataset}.py config_pytorch.py
python3 preprocessing/preprocessing.py
```

The script should have produced a folder in _resources/_ with the structure:
- {project_name}/:
  - data
    - video1
    - video2
    - ...
  - crops
    - video1
    - video2
    - ...
  - detections.csv

By default, the full video frame is used. In case you want to extract a specific ROI, modify the detection.csv file accordingly and run preprocessing.py again.
Finally, you can train the model and extract the features by running
```
python3 features/pytorch/train.py
python3 features/pytorch/extract_fc_features.py
python3 features/pytorch/extract_lstm_features.py
```


