from ast import parse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_probability as tfp

from datetime import datetime
from tqdm import tqdm

import os
import h5py
import argparse

from BalleHP import BalleHP
from read_data import read_data_numpy
import constants


def gpu_settings() -> tf.distribute.MirroredStrategy:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1']) # gpu distribution strategy
    return strategy

def read_data(ch_format='channels_first') -> np.ndarray:
    data_path = os.path.join(constants.DATA_FOLDER, constants.DATA_FILE)
    data      = read_data_numpy(data_path, ch_format)
    return data

def split_train_test(data: np.ndarray):
    train_images = data[:constants.TRAINING_SET_SIZE]
    test_images  = data[constants.TRAINING_SET_SIZE:constants.TRAINING_SET_SIZE+constants.VALIDATION_SET_SIZE]
    return train_images, test_images


def parse_args():
    parser = argparse.ArgumentParser()
    # norm arguments: if true the data is normalized to [0,1] (default: False)
    parser.add_argument('--norm', type=int, default=0)
    # epochs
    parser.add_argument('--epochs', type=int, default=constants.EPOCHS)
    return parser.parse_args()


def main():
    
    args   = parse_args()
    norm   = args.norm
    epochs = args.epochs
    
    if norm==1:
        norm_str = "normTrue"
    else:
        norm_str = "normFalse"
    
    strategy  = gpu_settings()
   
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    print('Reading data...')
    
    ch_format = 'channels_last' # channels_last !!!!
    data      = read_data(ch_format)
    if norm==1:
        print('Normalizing data to [0,1]...')
        data  = data.astype('float32') / 255.0
    elif norm==0:
        print('Data is not normalized to [0,1]...')
        data  = data.astype('float32')
        
    print('Data shape: {}'.format(data.shape))
    
    train_images, test_images = split_train_test(data)
    del data
    print('Train images shape: {}'.format(train_images.shape))

    buffer_size       = len(train_images) # buffer size for shuffling
    global_batch_size = constants.BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync # global batch size (in our case 2gpu * BATCH_SIZE_PER_REPLICA)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(global_batch_size)
    test_dataset  = tf.data.Dataset.from_tensor_slices(test_images).batch(global_batch_size)
    
    print('Distributing data...')
    train_dataset_dist = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset_dist  = strategy.experimental_distribute_dataset(test_dataset)

    del train_dataset, test_dataset
    
    
    with strategy.scope():
        
        vae       = BalleHP(N=128, M=192, k1=3, k2=3, c=3, format=ch_format)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 
        
        @tf.function
        def Loss(inputs, outputs):
            return tf.reduce_mean(tf.square(inputs - outputs[0])) + 0.5*(tf.reduce_mean(outputs[1])) + 0.5*(tf.reduce_mean(outputs[2]))

        @tf.function # compile the function to a graph for faster execution
        def train_step(inputs, vae):
            with tf.GradientTape() as tape: # create a tape to record operations
                outputs = vae(inputs) # forward pass
                loss = Loss(inputs, outputs) # MSE loss (maybe put this in a function)
            gradients = tape.gradient(loss, vae.trainable_variables) # compute gradients    
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables)) # gradient descent
            return loss # return loss for logging
        
        @tf.function
        def val_step(inputs, vae):
            outputs = vae(inputs, training=False) # forward pass
            loss    = Loss(inputs, outputs)       
            return loss 


    def train_step_dist(inputs, vae, strategy):
        loss = strategy.run(train_step, args=(inputs, vae))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    def val_step_dist(inputs, vae, strategy):
        loss = strategy.run(val_step, args=(inputs, vae))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        
    
    training_losses = []
    test_losses     = []
    
    print('Training started...')
    
    for epoch in range(epochs):
        total_loss  = 0.0 
        num_batches = 0 
        for inputs in tqdm(train_dataset_dist, 'training steps'): 
            total_loss  += train_step_dist(inputs, vae, strategy) # type: ignore # sum losses across replicas (replicas are the gpus
            num_batches += 1 # count number of batches
            
        train_loss = total_loss / num_batches # compute average loss
        training_losses.append(train_loss)
        
        print('Epoch {} train loss: {}'.format(epoch, train_loss))

        total_loss  = 0.0
        num_batches = 0
        for inputs in tqdm(test_dataset_dist, 'validation steps'): 
            total_loss  += val_step_dist(inputs, vae, strategy) # type: ignore # sum losses across replicas
            num_batches += 1 # count number of batches
            
        test_loss = total_loss / num_batches # compute average loss
        test_losses.append(test_loss)
        
        print('Epoch {} test loss: {}'.format(epoch, test_loss))
        
    print('Training finished!')
    
    print('Saving model and losses...')
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # save model
    model_name = f"model_hp_{ch_format}_epochs{epochs}_{norm_str}_{current_time}.h5"
    model_path = constants.MODEL_FOLDER + model_name
    vae.save_weights(model_path)
    
    # save losses in .h5 file
    losses_name = f"losses_hp_{ch_format}_epochs{epochs}_{norm_str}_{current_time}.h5"
    losses_path = constants.MODEL_FOLDER + losses_name
    with h5py.File(losses_path, 'w') as f: # type: ignore
        f.create_dataset('train', data=training_losses)
        f.create_dataset('test', data=test_losses)



if __name__ == '__main__':
    main()