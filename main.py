import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import tensorflow as tf

from trainer import Trainer
from plotter import Plotter

    
if __name__ == "__main__":
    trainer_args = {'loss':{'name':'mse'},
                    'metric':{'name':'Mean'},
                    'optimizer':{'name':'SGD','learning_rate':0.001},
                    'dataset':{'name':'uniform','batch_size':100,'epoch':1},
                    'model':{'name':'DNN','units':[64,16,1],
                             'activations':['tanh','tanh','tanh'],'fuse_models':1} }
    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    trainer.uniform_self_evaluate()

    plotter_args ={'num_evaluate': 10,
                    'step': 1/10,
                    'fuse_models': trainer_args['model']['fuse_models']}

    plotter = Plotter(plotter_args, trainer.model)
    plotter.plot_1d_loss(trainer=trainer)
    
