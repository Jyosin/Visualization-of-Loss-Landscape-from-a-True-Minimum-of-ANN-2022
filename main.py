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
                             'activations':['tanh','tanh','tanh'],'fuse_models':None} }
    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    trainer.uniform_self_evaluate()

    plotter_args ={'num_evaluate': 100,
                    'step': 1/10000,
                    'fuse_models': trainer_args['model']['fuse_models']}

    plotter = Plotter(plotter_args, trainer.model)

    fused_direction, normlized_direction = plotter.creat_random_direction(
        norm='layer')
    plotter.set_weights(init_state=True, init_directions=normlized_direction)

    start_time = time.time()
    for i in range(plotter.num_evaluate):
        plotter.set_weights(directions=[fused_direction])
        avg_loss = trainer.uniform_self_evaluate()
        with open("result_10000.csv", "ab")as f:
            np.savetxt(f, avg_loss, comments="")
        end_time = time.time()
        print("total time{}".format(end_time-start_time))
                            