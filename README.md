# mnist_cnn_cuda
Recognition of hand written numbers using Convolutional Neural Network implemented in Python (for reference) and in CUDA C++ on GPU device.

# Requirements
* *python3.6* (f-strings are used)
* *keras*
* *tensorflow* (backend for *keras*)

# Installation
It is recommended to use *virtualenv* or *docker* image for separating the project environment. 

## Clone the repository
```bash
git clone https://github.com/boczekbartek/mnist_cnn_cuda
cd mnist_cnn_cuda
```

## Basic
```bash
pip install --user -r requirements.txt
```

## Virtualenv
```bash
virtualenv -p python3.6 mnist_cnn_cuda_env
source mnist_cnn_cuda_env/bin/activate
pip install -r requirements.txt
```

## Docker
NOTE: you need to have Docker and docker-compose installed. See: (https://docs.docker.com/install/, 
https://docs.docker.com/compose/install/)
```bash
docker-compose build mnist-cpu
```

# Usage
## Models training
Train all available models run:
```bash
python mnist_py/train.py
```

List available models:
```bash
python mnist_py/train.py -h
```

Train one model:
```bash
python mnist_py/train.py small_cnn
```

Models definitions are stored inside *mnist_py/models* directory.

## Train with docker
All models
```bash
mkdir ~/.keras # MNIST will is cached here
docker-compose up mnist-cpu-train
```
Chosen models
```bash
MODELS='small_cnn' docker-compose up mnist-cpu-train
```

# Changelog
0.0.2
* Plotting training progress
* mnist_py: MLP model 
* mnist_py: Two CNN models: bigger, more complex and smaller, basic
* docker-compose improvements, local device keras cache use 
* Updated README.md

0.0.1 
* Dockerfile and docker-compose added
* Tensorflow and Keras - Python requirements
