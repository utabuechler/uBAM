This folder contains the necessary scripts for training the VAE and a Magnifier class used by other scripts for the magnification.

- To train the generative model, after filling the path in the [root]/config.py file, run

python [root]/magnification/training/main.py --gpu [GPU_ID]

- To apply the magnification on all sequences, run

python [root]/magnification/run_magnification/run_magnification.py --gpu [GPU_ID]


