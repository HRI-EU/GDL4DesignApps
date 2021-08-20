""" 
## LICENSE: GPL 3.0
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or 
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Algorithm for training the Vanilla 3D point cloud autoencoder used for the
research presented in:
- T. Rios, B. van Stein, S. Menzel, T. Bäck, B. Sendhoff, P. Wollstadt,
"Feature Visualization for 3D Point Cloud Autoencoders", 
International Joint Conference on Neural Networks (IJCNN), 2020
[https://www.honda-ri.de/pubs/pdf/4354.pdf]

- T. Rios, T. Bäck, B. van Stein, B. Sendhoff, S. Menzel, 
"On the Efficiency of a Point Cloud Autoencoder as a Geometric Representation
for Shape Optimization", 2019 IEEE Symposium Series on Computational Intelligence 
(SSCI), pp. 791-798, 2019.
[https://www.honda-ri.de/pubs/pdf/4199.pdf]

Pre-requisites/tested in:
 - Python      3.6.10
 - numpy       1.19.1
 - TensorFlow  1.14.0
 - TFLearn     0.3.2
 - cudatoolkit 10.1.168
 - cuDNN       7.6.5
 - Ubuntu      18.04
 - pandas      1.1.0
 - implementation of the Chamfer distance used in Achlioptas et al. (2017)
 [https://arxiv.org/abs/1707.02392], available at 
 https://github.com/optas/latent_3d_points. If the scripts are used in an
 environment with Python 2.7+, CUDA 8.0+ or cuDNN 6.0+, the scripts in the
 from Achlioptas et al. must be adjusted and compiled to the current versions
 in the environment.

Copyright (c)
Honda Research Institute Europe GmbH

Author: Thiago Rios <thiago.rios@honda-ri.de>
"""

# ==============================================================================
## Import Libraries
# General purpose
import os
import time
import sys
import argparse

# Mathematical / Scientific tools
import numpy as np
import pandas as pd
import random
import tensorflow as tf

# Ploting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Achlioptas original implementation
from latent_3d_points.external.structural_losses.tf_approxmatch import\
    approx_match, match_cost
from latent_3d_points.external.structural_losses.tf_nndistance import\
    nn_distance

# Autoencoder specific scripts
# - Data preprocessingfrom preproc_scripts import data_part_tree as DataPart
from preproc_scripts import data_set_norm, pointcloud_sampling
from preproc_scripts import data_part_tree as DataPart

# - Autoencoder architecture
from architecture_autoencoders import encoder_layer_gen, decoder_layer_gen,\
   ENCODER, DECODER

# ==============================================================================
### Auxiliary functions
## Standard plot of the losses
def LossesPlot(loss_training, loss_tests, filename):
    ''' Function for plotting the loss values during training
    Inputs:
      - loss_training: losses during training, shape = (-1,)
      - loss_tests: losses during tests, shape = (-1)
      - filename: name of the file for saving the plot, type = string
    Outputs:
      - save file according to the specified name (and location)
    '''
    loss_training = np.array(loss_training)
    loss_tests = np.array(loss_tests)
    epochs_array = np.array(range(len(loss_training)))
    fig = plt.figure()
    fig.set_size_inches(6, 2.7)
    plt.plot(epochs_array, loss_training[:,0],\
        'b--', linewidth=2, label="Training")
    plt.plot(epochs_array, loss_tests[:,0],\
        'r:', linewidth=2, label="Test")
    plt.xlabel("Epoch")#, font_dict={"size": 10})
    plt.xticks(fontsize=10)
    plt.ylabel("Loss function")#, font_dict={"size": 10})
    plt.xticks(fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return 

## Rotation operator to generate data augmentation
def Rotation(PC, theta, axis):
    ''' Rotation operator
    Reference: https://en.wikipedia.org/wiki/Rotation_matrix
    Inputs:
      - PC: Point Cloud, shape (-1, 3)
      - theta: angle of rotation, [rad]
      - axis: x, y, z
    Outputs:
      - P: rotated point cloud
    '''
    # Rotation around x
    if axis == 'x':
        P = np.matmul(\
            PC, [[1, 0, 0],\
                 [0, np.cos(theta), -np.sin(theta)],\
                 [0, np.sin(theta), np.cos(theta)]])

    # Rotation around y
    if axis == 'y':
        P = np.matmul(\
            PC, [[np.cos(theta), 0, np.sin(theta)],\
                 [0, 1, 0],\
                 [-np.sin(theta), 0, np.cos(theta)]])

    # Rotation around z
    if axis == 'z':
        P = np.matmul(\
            PC, [[np.cos(theta), -np.sin(theta), 0],\
                 [np.sin(theta), np.cos(theta), 0],\
                 [0, 0, 1]])
    return(P)

# ==============================================================================
### INITIALIZATION
## Get arguments from command line
parser = argparse.ArgumentParser(description='pc-ae training hyperparameters')
# Path to the data set
parser.add_argument('--i', type=str,\
        help='path to the data set')
# Path to the data set
parser.add_argument('--o', type=str,\
        help='path to output directory')
# Point cloud size
parser.add_argument('--N', type=int,\
        help='point cloud size')
# Size of the latent representation
parser.add_argument('--LR', type=int,\
        help='number of latent variables')
# GPU ID
parser.add_argument('--GPU', type=int,\
        help='GPU ID')
args = parser.parse_args()

# Assign GPU
os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(args.GPU))

## Clear screen (just for comfort)
os.system("clear")

## Seed for Random Number Generation
    # -- CAUTION! The seed controls how the random numbers are generated and
    #it guarantees the repeatability of the experiments
    # (generation of random shapes and initialization of the variables)
np.random.seed(seed=0)
random.seed(0)

# ==============================================================================
# SETTINGS SECTION
# ==============================================================================
### Data Management
## Dataset
# Directory where the point clouds are stored
pathToDataset = args.i
#'/hri/storage/visualdata/geometricdata/\
#ShapeNetCore.v2.refined/02958343'

# Output directory
name_exp = "pcae_N{}_LR{}".format(args.N, args.LR)
out_dir = args.o

# ==============================================================================
### Experimental setup
## Training Hyperparameters
# Size of the Point Clouds
pc_size = args.N
# Shuffle the order of the point clouds? (dataset, not ordering of the points)
shuffle_dataset = False
# Data Augmentation: number of random rotations around z. If not necessary, set
#to 0
data_augmentation = 0
# Batch size for training
training_batch_size = 50
# Batch size for testing
test_batch_size = 50

# Encoder layers: Number of features for each layer
  # An additional convolution layer is later added at [L325], such that  
  #the output of the encoder has the same dimensionality as the latent layer
encoder_layers = [64, 128, 128, 256]
# Latent Layer Size
latent_layer = args.LR 
# Decoder layers: Number of hidden neurons for each layer (per channel)
  # A fully connected layer is added at [L327], in order to set the
  #dimensionality of the output the same as the input
decoder_layers = [256, 256]

# Learning rate
l_rate = 5e-4
# Maximum number of Epochs
epochs_max = 700
# Stop Criteria
stop_training = 1e-5
# % of the dataset for training [0, 1]
frac_training = 0.9
## Rate Autosave
  # Number of elapsed epochs between saving logs
autosave_rate = 10

# ==============================================================================
### AUTOMATED SECTIONS
# ==============================================================================
### Generate output directory
if out_dir == None:
    top_dir = "Network_{}".format(name_exp)
else:
    top_dir = "{}/Network_{}".format(out_dir, name_exp)
if not os.path.exists(top_dir):
    os.makedirs(top_dir)
## Path to autoecoder graph files
pathToGraph = "{}/pcae".format(top_dir)

# ==============================================================================
### Load and assign point cloud data
# Loading the data set
dataset_full, log_names = pointcloud_sampling(pathToDataset, pc_size)

# Shuffling geometries
total_shapes = dataset_full.shape[0]
geom_sorting = list(range(total_shapes))
if shuffle_dataset: 
    random.shuffle(geom_sorting)
    dataset_full = dataset_full[geom_sorting,:,:] 
    log_names = np.array(log_names)[geom_sorting]

# Number of shapes for training
training_size = int(frac_training*total_shapes)
test_size = int(total_shapes - training_size)
# Change the test batch size, if necessary
if test_size < test_batch_size:
    print("**WARNING: Less shapes for testing than required in a batch")
    print("\tTest batch size modified to {}".format(test_size))
    test_batch_size = test_size

# Assign training set
training_set = dataset_full[0:training_size,:,:]
test_set = dataset_full[training_size:dataset_full.shape[0],:,:]
log_names_training = log_names[0:int(frac_training*total_shapes)]
log_names_test = log_names[int(frac_training*total_shapes):len(log_names)]

# Cleaning unnecessary variable
del dataset_full

## Data Augmentation
# Perform data augmentation by rotating the geometries around the z
#axis (vertical)
if data_augmentation > 0:
    Angle = np.random.random(data_augmentation)*1*np.pi - 0.5*np.pi
    training_set_aug = np.zeros(  # allocate memory for augmented data\
            (training_set.shape[0] * (data_augmentation + 1),\
             training_set.shape[1],\
             training_set.shape[2]))
    training_set_aug[:training_set.shape[0], :, :] = training_set
    ind = training_set.shape[0]
    for i in range(training_size):
        for j in range(data_augmentation):
            training_set_aug[ind, :, :] = np.reshape(Rotation(training_set[i,:,:], Angle[j], 2),\
                (1, -1, 3))
            ind = ind+1

    training_set = training_set_aug
    del training_set_aug

## Normalize data
data_set_n, norm_values = data_set_norm(np.concatenate(\
        (training_set, test_set), axis=0), np.array([0.1, 0.9]))
training_set = data_set_n[0:training_set.shape[0], :, :]
test_set = data_set_n[training_set.shape[0]:data_set_n.shape[0], :, :]

## Apply the data-partitioning tree algorithm to the dataset
training_set = DataPart(training_set, 3)[0]
test_set = DataPart(test_set, 3)[0]

## Saving log files
# Normalization limits
pd.DataFrame(norm_values).to_csv(str.format(\
    "{}/normvalues.csv", top_dir), header=None, index=None)
# Geometries in the training set
pd.DataFrame(log_names_training).to_csv(str.format(\
    "{}/geometries_training.csv", top_dir), header=None, index=None)
# Geometries in the test set
pd.DataFrame(log_names_test).to_csv(str.format(\
    "{}/geometries_testing.csv", top_dir), header=None, index=None)
    
# ==============================================================================
### Autoencoder Setup
# Add the last convolution layer
encoder_layers.append(latent_layer)
# Add the last fully connected layer
decoder_layers.append(pc_size)

## Assign encoder weights and bias, generate the input placeholders
x, conv_layers, bias_conv_layers = encoder_layer_gen(\
    encoder_layers, pc_size)
## Generate the encoder graph
latent_rep_full = ENCODER(x, conv_layers, bias_conv_layers, latent_layer)

## Assign decoder weights and bias
fc_layers, bias_fc_layers = decoder_layer_gen(latent_layer, decoder_layers)
## Generate the decoder graph
point_clouds = DECODER(latent_rep_full, decoder_layers, fc_layers,\
    bias_fc_layers)

# ==============================================================================
### Log the experiment setup
# Dictionary to save a log of the configuration used to run the experiment
experiment_setup = {}
# Input data
experiment_setup["dataset"] = pathToDataset
experiment_setup["shuffle_dataset"] = shuffle_dataset
experiment_setup["pc_size"] = pc_size
experiment_setup["data_augmentation"] = data_augmentation
experiment_setup["training_batch_size"] = training_batch_size
experiment_setup["test_batch_size"] = test_batch_size
## Autoencoder Architecture
experiment_setup["latent_layer"] = latent_layer
experiment_setup["encoder_layers"] = encoder_layers
experiment_setup["decoder_layers"] = decoder_layers
## Training the Autoencoder
experiment_setup["l_rate"] = l_rate
experiment_setup["epochs_max"] = epochs_max
experiment_setup["stop_training"] = stop_training
experiment_setup["frac_training"] = frac_training
experiment_setup["autosave_rate"] = autosave_rate

# Write file
f = open(str.format("{}/log_dictionary.py", top_dir), "w")
f.write("log_dictionary = ")
f.write(str(experiment_setup))
f.close()

# ==============================================================================
### Start Training
# Optimizer
optimizer = tf.train.AdamOptimizer(l_rate)
# Loss function: Chamfer distance
loss_a, _, loss_b, _ = nn_distance(x, point_clouds)
losses = tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b)
method = optimizer.minimize(losses)

# Initialize variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

# Calculate number of batches
batches_training = int(training_size*(1+data_augmentation)/training_batch_size)
batches_testing = int(test_size/test_batch_size)

print(batches_training, batches_testing)

# List for logging results
  # training losses
loss_values = []
  # test losses
test_loss = []

## Iteration over epochs
t = time.time()
for epochs in range(epochs_max):
    # temporary list for log
    loss_temp = []
        
    # Iteration over training batches
    for b in range(batches_training):
        # Assign batch
        minibatch = np.reshape(training_set[\
            b*training_batch_size:(b+1)*training_batch_size, :, :],\
                (training_batch_size, -1, 3))
        # Train 1 epoch over batch
        _, loss_val  = sess.run([method, losses], feed_dict={x: minibatch})
        # Log loss value
        loss_temp.append(loss_val)
    # Log mean and std over batches
    loss_values.append([np.mean(loss_temp), np.std(loss_temp)])

    # Iteration over test batches
    test_loss_temp = []
    for b in range(batches_testing):
        # Assign batch
        minibatch = np.reshape(test_set[\
            b*test_batch_size:(b+1)*test_batch_size, :, :],\
                (test_batch_size, -1, 3))
        # Test over one batch
        test_val = sess.run(losses, feed_dict={x: minibatch})
        # Log loss for a batch
        test_loss_temp.append(test_val)
    # Log mean and std over batches
    test_loss.append([np.mean(test_loss_temp), np.std(test_loss_temp)])
    
    # Generate logs, ploting and saving graph
    if epochs % autosave_rate == 0 and epochs > 0:
        # Elapsed time
        etime = time.time()-t
        timeh = np.floor(etime/3600)
        timemin = np.floor((etime/60 - timeh*60))
        timesec = (etime - timeh*3600 - timemin*60)
        # Remaining time
        rtime = (epochs_max-epochs)*(etime/epochs)
        rtimeh = np.floor(rtime/3600)
        rtimemin = np.floor((rtime/60 - rtimeh*60))
        rtimesec = (rtime - rtimeh*3600 - rtimemin*60)

        print(str.format("EPOCH: {}", epochs))
        print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
            "h",  str.format("{:.0f}", timemin), "min", \
                str.format("{:.0f}", timesec), "s")
        print("REMAINING TIME:", str.format("{:.0f}", rtimeh), \
            "h",  str.format("{:.0f}", rtimemin), "min", \
                str.format("{:.0f}", rtimesec), "s")

        # Report losses
        print(str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}", \
            np.mean(loss_temp), np.std(loss_temp)))
        print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}", \
            np.mean(test_loss_temp), np.std(test_loss_temp)))
        # Save log: losses
        pd.DataFrame(loss_values).to_csv(str.format(\
            "{}/{}_losses_training.csv", top_dir, name_exp), \
                header=None, index=None)
        pd.DataFrame(test_loss).to_csv(str.format(\
            "{}/{}_losses_test.csv", top_dir, name_exp), \
                header=None, index=None)

        # Plot losses
        plot_name = str.format("{}/plot_losses_{}.png", top_dir, name_exp)
        LossesPlot(loss_values, test_loss, plot_name)

        # Save Graph
        saver.save(sess, pathToGraph)
    if test_loss[-1][0] <= stop_training: 
        print("Stop criteria achieved!")
        print("Test CD: {:.3E}, crit.: {:.3E}".\
            format(test_loss[-1][0], stop_training))
        break

## FINISHING TRAINING
# Report losses
etime = time.time()-t
timeh = np.floor(etime/3600)
timemin = np.floor((etime/60 - timeh*60))
timesec = (etime - timeh*3600 - timemin*60)
print(str.format("EPOCH: {}", epochs))
print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
    "h",  str.format("{:.0f}", timemin), "min", \
        str.format("{:.0f}", timesec), "s")

print(str.format("EPOCH: {}", epochs))
print(str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}", \
    np.mean(loss_temp), np.std(loss_temp)))
print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}", \
    np.mean(test_loss_temp), np.std(test_loss_temp)))
# Save log: losses
pd.DataFrame(loss_values).to_csv(str.format(\
    "{}/{}_losses_training.csv", top_dir, name_exp), \
        header=None, index=None)
pd.DataFrame(test_loss).to_csv(str.format(\
    "{}/{}_losses_test.csv", top_dir, name_exp), \
        header=None, index=None)

# Plot losses
plot_name = str.format("{}/plot_losses_{}.png", top_dir, name_exp)
LossesPlot(loss_values, test_loss, plot_name)

# Save Graph
saver.save(sess, pathToGraph)
sess.close()
print("#### FINISHED!! ####")
exit()

#EOF
