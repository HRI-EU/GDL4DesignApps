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

## Scripts for interplating shapes in the latent space using the trained
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
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

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

## Autoencoder
# Graph information
if flagVAE: metaf = "vpcae"
else: metaf = "pcae"

pathToGraph = str.format("{}/{}", top_dir, metaf)
# - Import Graph at latest state (after training)
TFmetaFile = str.format("{}/{}.meta", top_dir, metaf)
TFDirectory = str.format("{}/", top_dir)

# ==============================================================================
### Extracting autoencoder paramters
## Directory containing the information about the autoencoder
out_dir = str.format(str.format("{}/network_param", top_dir))
if not osp.exists(out_dir): os.system("mkdir {}".format(out_dir))

## Start session
with tf.Session() as sess:
    # import graph data
    new_saver = tf.train.import_meta_graph(TFmetaFile, clear_devices=True)
    new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
    graph = tf.get_default_graph()

    # import network main layers
    # - Input
    x = graph.get_tensor_by_name("x:0")
    # encoder weights and bias
    w_conv_ten_d = {}
    b_conv_ten_d = {}
    for i in range(len(log_dictionary["encoder_layers"])):
        # Weights
        wname = "w_conv_layer_{}".format(i)
        wten = graph.get_tensor_by_name("{}:0".format(wname))
        wval = sess.run(wten)
        wval = np.reshape(wval, (wval.shape[1], wval.shape[2]))
        pd.DataFrame(wval).to_csv("{}/{}.dat".format(out_dir, wname), header=None, index=None)
        # Bias
        bname = "b_conv_layer_{}".format(i)
        bten = graph.get_tensor_by_name("{}:0".format(bname))
        bval = sess.run(bten)
        pd.DataFrame(bval).to_csv("{}/{}.dat".format(out_dir, bname), header=None, index=None)
        ## VAE: batch normalization parameters
        if flagVAE:
            if i == 0: 
                gname = "batch_normalization/gamma"
                btname = "batch_normalization/beta"
                mmname = "batch_normalization/moving_mean"
                mvname = "batch_normalization/moving_variance"
            else:
                gname = "batch_normalization_{}/gamma".format(i)
                btname = "batch_normalization_{}/beta".format(i)
                mmname = "batch_normalization_{}/moving_mean".format(i)
                mvname = "batch_normalization_{}/moving_variance".format(i)
    
            gten = graph.get_tensor_by_name("{}:0".format(gname))
            gval = sess.run(gten)
            pd.DataFrame(gval).to_csv("{}/batch_normalization_gamma_{}.dat".format(out_dir, i), header=None, index=None)
    
            btten = graph.get_tensor_by_name("{}:0".format(btname))
            btval = sess.run(btten)
            pd.DataFrame(btval).to_csv("{}/batch_normalization_beta_{}.dat".format(out_dir, i), header=None, index=None)
    
            mmten = graph.get_tensor_by_name("{}:0".format(mmname))
            mmval = sess.run(mmten)
            pd.DataFrame(mmval).to_csv("{}/batch_normalization_movmean_{}.dat".format(out_dir, i), header=None, index=None)
    
            mvten = graph.get_tensor_by_name("{}:0".format(mvname))
            mvval = sess.run(mvten)
            pd.DataFrame(mvval).to_csv("{}/batch_normalization_movvar_{}.dat".format(out_dir, i), header=None, index=None)
    
    # - Decoder weights and bias
    w_fc_ten_d = {}
    b_fc_ten_d = {}
    for i in range(len(log_dictionary["decoder_layers"])):
        # Weights
        wname = "w_fc_layer_{}".format(i)
        wten = graph.get_tensor_by_name("{}:0".format(wname))
        wval = sess.run(wten)
        pd.DataFrame(wval).to_csv("{}/{}.dat".format(out_dir, wname), header=None, index=None)
        # Bias
        bname = "b_fc_layer_{}".format(i)
        bten = graph.get_tensor_by_name("{}:0".format(bname))
        bval = sess.run(bten)
        pd.DataFrame(bval).to_csv("{}/{}.dat".format(out_dir, bname), header=None, index=None)

    sess.close()

exit()
# EOF
