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

##Scripts for interplating shapes in the latent space using the trained
parameters of the vanilla and variational 3D point cloud autoencoders
used for the research in:
- T. Rios, B. van Stein, S. Menzel, T. Bäck, B. Sendhoff, P. Wollstadt,
"Feature Visualization for 3D Point Cloud Autoencoders", 
International Joint Conference on Neural Networks (IJCNN), 2020
[https://www.honda-ri.de/pubs/pdf/4354.pdf]

- T. Rios, T. Bäck, B. van Stein, B. Sendhoff, S. Menzel, 
"On the Efficiency of a Point Cloud Autoencoder as a Geometric Representation
for Shape Optimization", 2019 IEEE Symposium Series on Computational Intelligence 
(SSCI), pp. 791-798, 2019.
[https://www.honda-ri.de/pubs/pdf/4199.pdf]

- S. Saha, S. Menzel, L.L. Minku, X. Yao, B. Sendhoff and P. Wollstadt, 
"Quantifying The Generative Capabilities Of Variational Autoencoders 
For 3D Car Point Clouds", 2020 IEEE Symposium Series on Computational Intelligence 
(SSCI), 2020. (submitted)

Pre-requisites:
 - Python      3.6.10
 - numpy       1.19.1
 - TensorFlow  1.14.0
 - TFLearn     0.3.2
 - cudatoolkit 10.1.168
 - cuDNN       7.6.5
 - Ubuntu      18.04
 - pandas      1.1.0
 - pyvista     0.29.1

Copyright (c)
Honda Research Institute Europe GmbH


Authors: Thiago Rios <thiago.rios@honda-ri.de>
"""

# ==============================================================================
## Import Libraries
# General purpose
import os
import os.path as osp
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
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# Achlioptas original implementation
from latent_3d_points.external.structural_losses.tf_approxmatch import\
    approx_match, match_cost
from latent_3d_points.external.structural_losses.tf_nndistance import\
    nn_distance

from preproc_scripts import data_part_tree as DataPart
from preproc_scripts import data_set_norm, pointcloud_sampling
from architecture_autoencoders import encoder_layer_gen, decoder_layer_gen,\
        vae_encoder_layer_gen, ENCODER, DECODER, vae_ENCODER, vae_DECODER

# ==============================================================================
### INITIALIZATION
## Get arguments from command line
parser = argparse.ArgumentParser(description='pc-ae training hyperparameters')
# List of shapes to be tested
parser.add_argument('--i', type=str,\
        help='path to the file with shapes to be tested')
# Point cloud size
parser.add_argument('--N', type=int,\
        help='point cloud size')
# Size of the latent representation
parser.add_argument('--LR', type=int,\
        help='number of latent variables')
# GPU ID
parser.add_argument('--GPU', type=int,\
        help='GPU ID')
# Variational autoencoder flag
parser.add_argument('--VAE', type=str,\
        help='Flag VAE')
args = parser.parse_args()

# Assign GPU
os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(args.GPU))

## check VAE Flag
if args.VAE == None:
    flagVAE = False
elif args.VAE.lower() == "true":
    flagVAE = True
else:
    flagVAE = False

## Clear screen (just for comfort)
os.system("clear")

## Seed for Random Number Generation
    # -- CAUTION! The seed controls how the random numbers are generated and
    #it guarantees the repeatability of the experiments
    # (generation of random shapes and initialization of the variables)
np.random.seed(seed=0)
random.seed(0)

# ==============================================================================
### Settings
## Name of the Experiment
name_exp = "pcae_N{}_LR{}".format(args.N, args.LR)
if flagVAE: name_exp = "v_"+name_exp

# ==============================================================================
### Preprocessing
## Directory contatining the information about the autoencoder
top_dir = str.format(str.format("Network_{}", name_exp))
if not osp.exists(top_dir):
    print(str.format("Directory {} does not exist!", top_dir))
    exit()

# Load log dictionary with the training and network information
os.system("cp {}/log_dictionary.py .".format(top_dir))
from log_dictionary import log_dictionary
os.system("rm log_dictionary.py")

## List of shapes to reconstruct
path_to_list=args.i
if not os.path.exists(path_to_list):
    print("***ERROR! Path to list of test data not found.")
    print("Script interrupted.")
    exit()

## Geometry names
geom_testing = np.array(pd.read_csv(path_to_list, header=None))

## Autoencoder
# Graph information
if flagVAE: metaf = "vpcae"
else: metaf = "pcae"

pathToGraph = str.format("{}/{}", top_dir, metaf)
# - Import Graph at latest state (after training)
TFmetaFile = str.format("{}/{}.meta", top_dir, metaf)
TFDirectory = str.format("{}/", top_dir)

# ==============================================================================
### Loading the shapes
## Number of points in the point cloud
pc_size = log_dictionary["pc_size"]

## Number of shapes to be reconstructed
n_shapes = geom_testing.shape[0]

## Sampling point clouds
# Initialize the batch for assigning the point clouds
vis_set = np.zeros((n_shapes, pc_size, 3))
cntr = 0
# Assign test set
for i in range(n_shapes):
    pc_load = np.array(\
        pd.read_csv(geom_testing[i][0], header=None, delimiter=" "))[:,0:3]
    if pc_load.shape[0] < pc_size:
        pc_sample = np.random.choice(\
            np.array(range(pc_load.shape[0])).flatten(), pc_size)
    else:
        pc_sample = np.random.choice(\
            np.array(range(pc_load.shape[0])).flatten(), pc_size, replace=False)
    vis_set[cntr,:,:] = pc_load[pc_sample,:]
    cntr = cntr+1

## Normalize the set of geometries
# Load point cloud normalization values
normDS = np.array(pd.read_csv(str.format("{}/normvalues.csv", top_dir),\
    header=None)).flatten()
maX = normDS[0]
miX = normDS[1]
Delta = maX - miX
# Normalize the data
vis_set,_ = data_set_norm(vis_set, np.array([0.1, 0.9]), \
        inp_lim=np.array([miX, maX]))

# ==============================================================================
### Reconstruction loss
## Create directory to save the files
dir_pc_plot = str.format("{}/pcae_test", top_dir)
os.system(str.format("mkdir {}", dir_pc_plot))

## Create file to log results
rec_losses = np.array(np.zeros((vis_set.shape[0], 2+args.LR)), dtype="U70")
rec_losses[:,0] = geom_testing[:,0]

## Start session
with tf.Session() as sess:
    # import graph data
    new_saver = tf.train.import_meta_graph(TFmetaFile, clear_devices=True)
    new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
    graph = tf.get_default_graph()
    # import network main layers
    # - Input
    x = graph.get_tensor_by_name("x:0")
    # - Latent Representation
    latent_rep = graph.get_tensor_by_name("latent_rep:0")
    # - Point clouds
    point_clouds = graph.get_tensor_by_name("PC:0")

    ## Tensor of reconstruction losses
    # Loss function: Chamfer distance
    loss_a, _, loss_b, _ = nn_distance(x, point_clouds)
    losses = tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b)
    
    # Dropout rate (for VAE only)
    if flagVAE: dpout = graph.get_tensor_by_name("do_rate:0")

    ## Calculate reconstruction losses and latent representations
    for i in range(n_shapes):
        pc = np.reshape(vis_set[i,:,:], (1, -1, 3))
        if flagVAE:
            lr, loss = sess.run([latent_rep, losses],\
                feed_dict={x: pc, dpout: 1.0})
        else:
            lr, loss = sess.run([latent_rep, losses],\
                feed_dict={x: pc})

        # Log results
        rec_losses[i, 1:1+args.LR] = lr.flatten()
        rec_losses[i, 1+args.LR:] = loss

    sess.close()

# Write file with results
header_f = ["shape",]
for i in range(args.LR): header_f += ["lr_{}".format(i)]
header_f += ["chamfer_distance"]
pd.DataFrame(rec_losses).to_csv("{}/reconstruction_losses.dat".\
    format(dir_pc_plot),\
    header=header_f, index=None)

exit()
# EOF
