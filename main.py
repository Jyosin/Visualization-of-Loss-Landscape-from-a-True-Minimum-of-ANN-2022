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
                             'activations':['tanh','tanh','tanh'],'fuse_models':10} }
    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    trainer.uniform_self_evaluate()

    # plotter_args ={''

    # }

    # plotter = Plotter(trainer.model)
    # normlized_random_direction = plotter.creat_random_direction(norm='layer')

    # N = 1000
    # step = 1/100

    # plotter.set_weights([normlized_random_direction], step=-step+N/2)

    # start_time = time.time()
    # for i in range(N):
    #     plotter.set_weights([normlized_random_direction], step)
    #     avg_loss = trainer.uniform_self_evaluate()
    #     with open("result_10000.csv", "ab")as f:
    #         np.savetxt(f, [avg_loss], comments="")
    #     end_time = time.time()
    #     print("total time{}".format(end_time-start_time))
                            