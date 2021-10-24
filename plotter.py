import os
import tensorflow as tf

from models import DNN
from data_generator import read_data_from_csv

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
            weight.assign_add(change)

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
            normalized_direction = direction * weights
        elif norm == 'd_filter':
            normalized_direction = []
            for d in direction :
                normalized_direction.append(d/ (tf.norm(d)+1e-10))
        elif norm == 'd_layer':
            normalized_direction = direction/tf.norm(direction)
        return normalized_direction

    def normalize_directions_for_weights(self,direction,weights, norm="filter", ignore="bias_bn"):
        normalized_direction = []
        for d,w in zip(direction,weights):
            if len(d.shape) <= 1:
                if ignore == "bias_bn":
                    d = tf.zeros(d.shape)
                else:
                    d=w
                normalized_direction.append(d)
            else:
                normalized_direction.append(
                    self.normalize_direction(d ,w, norm))
        return normalized_direction

    def create_target_direction(self):
        pass

    def creat_random_direction(self, ignore='bias_bn', norm='filter'):
        weights = self.get_weights()
        direction = self.get_random_weights(weights)
        direction = self.normalize_directions_for_weights(
            direction,weights, norm ,ignore
        )
        return direction
        

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
                    'dataset':{'name':'uniform','batch_size':100,'epoch':1},
                    'model':{'name':'DNN','units':[64,16,1],
                             'activations':['tanh','tanh','tanh']}, }

    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    trainer.self_evaluate()

    # plotter = Plotter(trainer.model)
    # normalized_random_direction = plotter.creat_random_direction(norm='layer')

    # N = 1000
    # step = 1/100
    # plotter.set_weights([normalized_random_direction],step=-step*N/2)
    
    # # plotter.set_weights([normalized_random_direction], step=0.5)

    # for i in range(N):
    #     plotter.set_weights([normalized_random_direction], step=step)
    #     avg_loss = trainer.self_eveliate()
    #     with open("result_1000.csv","ab") as f:
    #         np.savetxt(f, [avg_loss], comments="")