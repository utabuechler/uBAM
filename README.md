Here we collect all our code for the nature methods journal.

#System requirements
The software was tested on `Ubuntu 18.04`.
The code runs on Python 3. We suggest to instal the newest version of Anaconda.

##Dependencies
You can install all required dependencies using the following command.
```
pip install -r requirements.txt
```

### CPU-only dependencies
The previous command automatically installs the gpu version of pytorch which is necessary for training the model. The demo does not need to run on a GPU, since uses a pretrained model. Therefore you can install the cpu version of pytorch running the following command
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
Since the software is developed in python, there is no need to compile. Therefore the installation time is the same needed to install the dependences and download the data. This should not take more than one hour, but it strongly depends on the system status and internet speed.

#Demo
##How to run
##Output
##Runtime

#Run on your data

