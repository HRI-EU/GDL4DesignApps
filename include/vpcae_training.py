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

## Algorithm for training the variational 3D point cloud autoencoder used for
the research presented in:
- S. Saha, S. Menzel, L.L. Minku, X. Yao, B. Sendhoff and P. Wollstadt, 
"Quantifying The Generative Capabilities Of Variational Autoencoders 
For 3D Car Point Clouds", 2020 IEEE Symposium Series on Computational Intelligence 
(SSCI), 2020. (submitted)

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

Author: Sneha Saha <sneha.saha@honda-ri.de>
"""

# ==============================================================================
## Import Libraries
# General purpose
import os
import time
import sys
import argparse
import pickle

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
from latent_3d_points.external.structural_losses.tf_approxmatch import approx_match, match_cost
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance

# Autoencoder specific scripts
# - Data preprocessing
from preproc_scripts import data_part_tree as DataPart
from preproc_scripts import data_set_norm, pointcloud_sampling
from architecture_autoencoders import vae_encoder_layer_gen, decoder_layer_gen, \
    vae_ENCODER, vae_DECODER

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
    plt.plot(epochs_array, loss_training[:, 0], \
             'b--', linewidth=2, label="Training")
    plt.plot(epochs_array, loss_tests[:, 0], \
             'r:', linewidth=2, label="Test")
    plt.xlabel("Epoch")  # , font_dict={"size": 10})
    plt.xticks(fontsize=10)
    plt.ylabel("Loss function")  # , font_dict={"size": 10})
    plt.xticks(fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return

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
name_exp = "v_pcae_N{}_LR{}".format(args.N, args.LR)
out_dir = args.o

# ==============================================================================
### Experimental setup
## Training Hyperparameters
# Size of the Point Clouds
pc_size = args.N
# Shuffle the order of the point clouds? (dataset, not ordering of the points)
shuffle_dataset = True
# Data Augmentation: number of random rotations around z. If not necessary, set
#to 0
data_augmentation = 0
# Batch size for training
training_batch_size = 10#80
# Batch size for testing
test_batch_size = 10#80

# Encoder layers: Number of features for each layer
  # An additional convolution layer is later added at [L294], such that  
  #the output of the encoder has the same dimensionality as the latent layer
encoder_layers = [64, 128, 128, 256]
# Latent Layer Size
latent_layer = args.LR 
# Decoder layers: Number of hidden neurons for each layer (per channel)
  # A fully connected layer is added at [L297], in order to set the
  #dimensionality of the output the same as the input
decoder_layers = [256, 256]

# Learning rate
l_rate = 5e-3
# Keep rate for dropout at the last convolution layer
dropout_conv = 1.0
## Parameters for balancing the loss functions
# alpha: Chamfer distance
alpha = 250
# beta: KL-divergence
beta = 0.001
# Maximum number of Epochs
epochs_max = 700
# Stop Criteria
stop_training = 1e-5
# % of the data set for training [0, 1]
frac_training = 0.85
# % of the data set for validation [0, 1]
frac_validation = 0.05
## Rate Autosave
  # Number of elapsed epochs between saving logs
autosave_rate = 100

# ==============================================================================
### AUTOMATED SECTIONS
# ==============================================================================
### Experimental setup
## Generate output directory
if out_dir == None:
    top_dir = "Network_{}".format(name_exp)
else:
    top_dir = "{}/Network_{}".format(out_dir, name_exp)
if not os.path.exists(top_dir):
    os.makedirs(top_dir)
## Path to autoecoder graph files
pathToGraph = "{}/vpcae".format(top_dir)

# ==============================================================================
### Load and assign point cloud data
# Loading the data set
dataset_full, log_names = pointcloud_sampling(pathToDataset, pc_size)

# Shuffling geometries
total_shapes = dataset_full.shape[0]
geom_sorting = list(range(total_shapes))
if shuffle_dataset:
    random.shuffle(geom_sorting)
    dataset_full = dataset_full[geom_sorting, :, :]
    log_names = np.array(log_names)[geom_sorting]

# Normalize the data set
dataset_full, norm_values = data_set_norm(dataset_full, np.array([0.1, 0.9]))

# Number of shapes for training
training_size = int(frac_training * total_shapes)
validation_size = int(frac_validation * total_shapes)
test_size = int(total_shapes - (training_size + validation_size))

## Assign training, validation and test set
# Point clouds
training_set = dataset_full[0:training_size, :, :]
validation_set = dataset_full[training_size : (training_size+validation_size), :, :]
test_set = dataset_full[(training_size+validation_size):dataset_full.shape[0], :, :]
# File names
log_names_training = log_names[0:int(frac_training * total_shapes)]
log_names_validation = log_names[\
    int(frac_training * total_shapes):int((frac_training + frac_validation) * total_shapes)]
log_names_test = log_names[int((frac_training+frac_validation) * total_shapes):len(log_names)]

# Cleaning unnecessary variable
del dataset_full

## Re-shuffling dataset
if shuffle_dataset:
    indices_shuffle2 = list(range(training_set.shape[0]))
    random.shuffle(indices_shuffle2)
    training_set = training_set[indices_shuffle2, :, :]

## Apply the data-partitioning tree algorithm to the data set
training_set = DataPart(training_set, 3)[0]
validation_set = DataPart(validation_set, 3)[0]
test_set = DataPart(test_set, 3)[0]

## Saving log files
# Normalization limitsshuffle_dataset = True
pd.DataFrame(norm_values).to_csv(str.format( \
    "{}/normvalues.csv", top_dir), header=None, index=None)
# Geometries in the training set
pd.DataFrame(log_names_training).to_csv(str.format( \
    "{}/geometries_training.csv", top_dir), header=None, index=None)
pd.DataFrame(log_names_validation).to_csv(str.format( \
    "{}/geometries_validation.csv", top_dir), header=None, index=None)
# Geometries in the test set
pd.DataFrame(log_names_test).to_csv(str.format( \
    "{}/geometries_testing.csv", top_dir), header=None, index=None)

## Saving the training, test and validation set in pickle file
# Training set
train_pickle= str.format("{}/training_set.pkl", top_dir)
with open(train_pickle, 'wb') as f:
    pickle.dump(training_set, f)
# Validation set
validation_pickle= str.format("{}/validation_set.pkl", top_dir)
with open(validation_pickle, 'wb') as f:
    pickle.dump(validation_set, f)
# Test set
test_pickle= str.format("{}/test_set.pkl", top_dir)
with open(test_pickle, 'wb') as f:
    pickle.dump(test_set, f)

# ==============================================================================
### Variational Autoencoder Setup
# Add the last convolution layer
encoder_layers.append(latent_layer)
encoder_layers.append(latent_layer)
# Add the last fully connected layer
decoder_layers.append(pc_size)

## Assign encoder weights and bias, generate the input placeholders
x, keep_prob, conv_layers, bias_conv_layers = vae_encoder_layer_gen( \
    encoder_layers, pc_size)
## Generate the encoder graph
latent_rep_full, z_mu, z_log_sigma = vae_ENCODER(x, conv_layers, \
    bias_conv_layers, latent_layer, keep_prob)

## Assign decoder weights and bias
fc_layers, bias_fc_layers = decoder_layer_gen(latent_layer, decoder_layers)
## Generate the decoder graph
point_clouds = vae_DECODER(latent_rep_full, decoder_layers, fc_layers, \
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
experiment_setup["frac_validation"] = frac_validation
experiment_setup["autosave_rate"] = autosave_rate
experiment_setup["dropout_conv"] = dropout_conv
experiment_setup["alpha"] = alpha
experiment_setup["beta"] = beta

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
Loss_cd_a, _, Loss_cd_b, _ = nn_distance(x, point_clouds)
losses_cd = tf.reduce_mean(Loss_cd_a) + tf.reduce_mean(Loss_cd_b)
recon_loss = tf.math.scalar_mul(alpha, losses_cd)
# KL divergence
KL_div_loss = - 0.5 * tf.reduce_sum(1 + z_log_sigma - tf.pow(z_mu, 2) - \
    tf.exp(z_log_sigma), reduction_indices=1)
KL_div_loss = tf.reduce_mean(KL_div_loss)
KL_loss = tf.math.scalar_mul(beta, KL_div_loss)
# Combined losses
losses_total = tf.add(recon_loss, KL_loss)
method = optimizer.minimize(losses_total)

# Initialize variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

# Calculate number of batches
batches_training = int(training_size * (1 + data_augmentation) / training_batch_size)
batches_testing = int(test_size / test_batch_size)

# List for logging results
  # total training losses per epoch
loss_values = []
  # KL-divergence parcel of training losses
loss_values_KL = []
  # CD parcell of test losses
loss_values_recon = []
  # ?
train_mu = []
  # ?
train_sigma = []
  # total test losses per epoch
test_loss = []
  # KL-divergence parcel of test losses
test_loss_KL =[]
  # CD parcel of test losses
test_loss_recon = []
  # ?
test_mu = []
  # ?
test_sigma = []

## Iteration over epochs
t = time.time()
for epochs in range(epochs_max):
    # temporary list for log
    loss_temp = []
    loss_temp_KL = []
    loss_temp_recon = []
    temp_mu = []
    temp_sigma = []

    # Iteration over training batches
    for b in range(batches_training):
        # Assign batch
        minibatch = np.reshape(training_set[\
                    b*training_batch_size:(b+1)*training_batch_size, :, :],\
                    (training_batch_size, -1, 3))
        # Train 1 epoch over batch
        _, loss_val_KL, loss_val_recon, loss_val = sess.run(\
            [method, KL_div_loss, losses_cd, losses_total], feed_dict={\
            x: minibatch, keep_prob: dropout_conv})
        # Log loss for a batch
        loss_temp.append(loss_val)
        loss_temp_KL.append(loss_val_KL)
        loss_temp_recon. append(loss_val_recon)
    # Log mean and std over batches
    loss_values.append([np.mean(loss_temp), np.std(loss_temp)])
    loss_values_KL.append([np.mean(loss_temp_KL), np.std(loss_temp_KL)])
    loss_values_recon.append([np.mean(loss_temp_recon), np.std(loss_temp_KL)])

    # Iteration over test batches
    test_loss_temp = []
    test_loss_temp_KL = []
    test_loss_temp_recon =[]
    for b in range(batches_testing):
        # Assign batch
        minibatch = np.reshape(test_set[ \
                    b * test_batch_size:(b + 1) * test_batch_size, :, :], \
                    (test_batch_size, -1, 3))
        # Test over one batch
        test_val = sess.run([KL_div_loss, losses_cd, losses_total], feed_dict={\
            x: minibatch, keep_prob: 1.0})
        # Log loss for a batch
        test_loss_temp.append(test_val[2])
        test_loss_temp_KL.append(test_val[0])
        test_loss_temp_recon.append(test_val[1])
    # Log mean and std over batches
    test_loss.append([np.mean(test_loss_temp), np.std(test_loss_temp)])
    test_loss_KL.append([np.mean(test_loss_temp_KL), np.std(test_loss_temp_KL)])
    test_loss_recon.append([np.mean(test_loss_temp_recon), np.std(test_loss_temp_recon)])

    # Generate logs, ploting and saving graph
    if epochs % autosave_rate == 0 and epochs > 0:
        # Elapsed time
        etime = time.time() - t
        timeh = np.floor(etime / 3600)
        timemin = np.floor((etime / 60 - timeh * 60))
        timesec = (etime - timeh * 3600 - timemin * 60)
        print(str.format("EPOCH: {}", epochs))
        print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
              "h", str.format("{:.0f}", timemin), "min", \
              str.format("{:.0f}", timesec), "s")

        # Report losses
        print(str.format("\t- Training :: Losses = {} +/- {}", \
                         np.mean(loss_temp), np.std(loss_temp)))
        print(str.format("\t- KL_Training :: Losses = {} ", \
                         np.mean(loss_temp_KL)))
        print(str.format("\t- Recon  :: Losses = {} ", \
                         np.mean(loss_values_recon)))
        print(str.format("\t- Test :: Losses = {} +/- {}", \
                         np.mean(test_loss_temp), np.std(test_loss_temp)))
        print(str.format("\t- KL_Test :: Losses = {} ", \
                         np.mean(test_loss_temp_KL)))
        # Save log: losses
        pd.DataFrame(loss_values).to_csv(str.format( \
            "{}/{}_losses_training.csv", top_dir, name_exp), \
            header=None, index=None)
        pd.DataFrame(test_loss).to_csv(str.format( \
            "{}/{}_losses_test.csv", top_dir, name_exp), \
            header=None, index=None)

        #KL divergeence loss
        pd.DataFrame(loss_values_KL).to_csv(str.format( \
            "{}/{}_KL_losses_training.csv", top_dir, name_exp), \
            header=None, index=None)

        pd.DataFrame(test_loss_KL).to_csv(str.format( \
            "{}/{}_KL_losses_test.csv", top_dir, name_exp), \
            header=None, index=None)

        # saving the reconstruction loss
        pd.DataFrame(loss_values_recon).to_csv(str.format( \
            "{}/{}_recon_losses_training.csv", top_dir, name_exp), \
            header=None, index=None)
        pd.DataFrame(test_loss_recon).to_csv(str.format( \
            "{}/{}_recon_losses_test.csv", top_dir, name_exp), \
            header=None, index=None)


        # Plot losses
        plot_name = str.format("{}/plot_losses_{}.png", top_dir, name_exp)
        LossesPlot(loss_values, test_loss, plot_name)

        # Save Graph
        saver.save(sess, pathToGraph)

## FINISHING TRAINING
# Report losses
etime = time.time() - t
timeh = np.floor(etime / 3600)
timemin = np.floor((etime / 60 - timeh * 60))
timesec = (etime - timeh * 3600 - timemin * 60)
print(str.format("EPOCH: {}", epochs))
print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
      "h", str.format("{:.0f}", timemin), "min", \
      str.format("{:.0f}", timesec), "s")

print(str.format("EPOCH: {}", epochs))
print(str.format("\t- Training :: Losses = {} +/- {}", \
                 np.mean(loss_temp), np.std(loss_temp)))
print(str.format("\t- Test :: Losses = {} +/- {}", \
                 np.mean(test_loss_temp), np.std(test_loss_temp)))
# Save log: losses
pd.DataFrame(loss_values).to_csv(str.format( \
    "{}/{}_losses_training.csv", top_dir, name_exp), \
    header=None, index=None)
pd.DataFrame(test_loss).to_csv(str.format( \
    "{}/{}_losses_test.csv", top_dir, name_exp), \
    header=None, index=None)


pd.DataFrame(loss_values_KL).to_csv(str.format( \
            "{}/{}_KL_losses_training.csv", top_dir, name_exp), \
            header=None, index=None)
pd.DataFrame(test_loss_KL).to_csv(str.format( \
            "{}/{}_KL_losses_test.csv", top_dir, name_exp), \
            header=None, index=None)

# saving the reconstruction loss
pd.DataFrame(loss_values_recon).to_csv(str.format( \
    "{}/{}_recon_losses_training.csv", top_dir, name_exp), \
    header=None, index=None)
pd.DataFrame(test_loss_recon).to_csv(str.format( \
    "{}/{}_recon_losses_test.csv", top_dir, name_exp), \
    header=None, index=None)

# Plot losses
plot_name = str.format("{}/plot_losses_{}.png", top_dir, name_exp)
LossesPlot(loss_values, test_loss, plot_name)

# Save Graph
saver.save(sess, pathToGraph)
sess.close()
print("#### FINISHED!! ####")
exit()

# EOF
