#!/usr/bin/env python3.6
""" See number of parameters of available models """
from train import models_initializers

if __name__ == '__main__':
    for model_name, model_init in models_initializers.items():
        print(f'{model_name}: {model_init().count_params()}')
