import os
import h5py
from numpy.lib.function_base import _parse_input_dimensions
import tensorflow as tf

from trainer import Trainer

class Plotter:
    def __init__(self, model):
        self.model = model


    def get_weights(self):
        return self.model.trainable_weights
    
    def set_weights(self,directions=None, step=None):
        #l(alpha * theta + (1- alpha)* theta')=> L(theta + alpha *(theta-theta'))
        #l(theta + alpha * theta_1 + beta * theta_2)
        #Each direction have same shape with trainable weights
        #)
        if directions == None:
            print("None of directions")
        else:
            if len(directions) == 2:
                dx = directions[0]
                dy = directions[1]
                changes = [step[0]*d0 +step[1]*d1 for (d0, d1)in zip(dx, dy)]
            else:
                changes = [d*step for d in directions[0]]

        weights = self.get_weights()
        for(weight,change) in zip(weights, changes):
            weight +=change

    def get_random_weights(self,weights):
        return [tf.random.normal(w.shape)for w in weights]

    def get_diff_weights(self, weights_1, weights_2):
        return [w2 - w1 for (w1, w2) in zip(weights_1, weights_2)]

    def normalize_direction(self,direction,weights,norm='filter'):
        # filter normalize : d = direction / norm(direction) * weight
        if norm == 'filter':
            normalized_direction = []
            for d, w in zip(direction, weights):
                normalized_direction.append(
                d = d / (tf.norm(d) + 1e-10) * tf.norm(w))
        elif norm == 'layer':
            normalized_direction = direction * tf.norm(weights)/tf.norm(direction)
        elif norm == 'weight':
            
        elif norm == 'd_filter':
            pass
        elif norm == 'd_layer':
            pass

    def normlize_direction_for_weights(self,direction,weights, norm="filter", ignore="bias_ 212")
        for d,w in zip(direction,weights):
            if ignore == "bias_bn":
                d = tf.zeros(d.shape)
    def create_target_direction(self):
        pass

    def creat_random_direction(self):
        pass

    def setup_direction(self):
        pass

    def name_direction_file(self):
        pass

    def load_directions(self):
        pass

if __name__ == "__main__":
    trainer_args = {'loss':{'name':'mse'},
                    'metric':{'name':'Mean'},
                    'optimizer':{'name':'SGD','learning_rate':0.001},
                    'dataset':{'name':'uniform','batch_size':12,'epoch':3},
                    'model':{'name':'DNN','units':[64,16,1],
                             'activations':['tanh','tanh','tanh']}, }