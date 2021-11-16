import sys
sys.path.append('..')

from utils import print_error
from data_generator import read_data_from_csv
from .base_trainer import BaseTrainer
import tensorflow as tf

class UniformTrainer(BaseTrainer):
    def __init__(self, args) :
        super(UniformTrainer, self).__init__(args=args)

    def _build_dataset(self, dataset_args) :
        self.x_v = None
        self.y_v = None
        dataset = read_data_from_csv(filename=dataset_args['path_to_data'],
                                        batch_size=dataset_args['batch_size'],
                                        CSV_COLUMNS=['x','y'],
                                        num_epochs=dataset_args['epoch'])

        return dataset


    def _just_build(self):
        try:
            iter_ds = iter(self.dataset)
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'],(-1,1))
            self.model(x['x'])
        except:
            print_error("build model with variables failed.")

    def train_step(self,x):
        inputs = x['x']
        labels = x['y']
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss(prediction,labels)
            grad = tape.gradient(loss,self.model.trainable_variables)
        
        
        self.optimizer.apply_gradients(zip(grad,self.model.trainable_variables))
        self.metric.update_state(loss)

    def run(self):
        iter_ds = iter(self.dataset)

        while True : 
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'],(-1,1))
            x['y'] = tf.reshape(x['x'],(-1,1))

            self.train_step(x)
            print("loss:",self.metric.result().numpy())
            self.metric.reset_states()

    def save_model_weights(self,filepath='./saved_models',name = 'model.h5',save_format="h5"):
        num = len(os.listdir(filepath))
        save_path = os.path.join(filepath,str(num)+'/')
        if os.path.exists(save_path):
            self.model.save_weights(save_path+name, save_format=save_format)
        else:
            os.mkdir(save_path)
            self.model.save_weights(save_path+name, save_format=save_format)
        print("model saved in {}".format(save_path+name))

    def load_model_weights(self,filepath='./saved_models',num = -1,name='model.h5'):
        if num == -1:
            num = len(os.listdir(filepath))-1
        filepath = os.path.join(filepath,str(num)+'/')

        if os.path.exists(filepath):
            # import pdb
            # pdb.set_trace()
            self.just_build()
            self.model.load_weights(filepath+name)
            print("model load from {}".format(filepath+name))
        else:
            print("path dosen't exits.")

    def evaluate_in_all(self, inputs, labels):
        prediction = self.model(inputs)
        loss = self.loss(prediction, labels)
        if self.args['model']['fuse_models'] == None:
            self.metric.update_state(loss)
            avg_loss = self.metric.result()
        else:
            avg_loss = tf.reduce_mean(loss, axis=-1)
        return avg_loss

    def uniform_self_evaluate(self, percent=20):
        # import pdb
        # pdb.set_trace()
        iter_test =iter(self.dataset)
        self.metric.reset_states()
        
        all_x =[]
        all_y =[]
        if isinstance(self.x_v,type(None)):
            while True and percent != 0:
                try:
                    x= iter_test.get_next()
                    x['x'] = tf.reshape(x['x'],(-1,1))
                    x['y'] = tf.reshape(x['y'],(-1,1))
                    all_x.append(x['x'])
                    all_y.append(x['y'])
                    percent -= 1
                except:
                    print("run out of data")
                    break
            self.x_v = tf.concat(all_x, axis=0)
            self.y_v = tf.concat(all_y, axis=0)

        avg_loss = self.evaluate_in_all(self.x_v, self.y_v)
        avg_loss = tf.reshape(avg_loss, shape=(-1,1))
        np_avg_loss = avg_loss.numpy()
        # print("Avg loss", np_avg_loss)
        return np_avg_loss



    def self_evaluate(self):
        iter_test = iter(self.dataset)
        self.metric.reset_states()
        FLAG=1
        while True:
            print("i am running {}".format(FLAG))
            FLAG+=1
            try:
                x = iter_test.get_next()
            except:
                print("run out of data. ")
                break
            x['x'] = tf.reshape(x['x'],(-1,1))
            x['y'] = tf.reshape(x['y'],(-1,1))
            prediction = self.model(x['x'])

            loss = self.loss(prediction,x['y'])
            self.metric.update_state(loss)
        
        avg_loss = self.metric.result().numpy()
        print("Avg loss", avg_loss)
        return avg_loss

if __name__ == "__main__":
    trainer_args = {'loss':{'name':'mse'},
                    'metric':{'name':'Mean'},
                    'optimizer':{'name':'SGD','learning_rate':0.001},
                    'dataset':{'name':'uniform','batch_size':12,'epoch':3},
                    'model':{'name':'DNN','units':[64,16,1],
                             'activations':['tanh','tanh','tanh']}, }
    
    trainer = Trainer(trainer_args)
    trainer.load_model_weights()