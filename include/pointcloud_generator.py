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
# List of shapes to be tested
parser.add_argument('--i', type=str,\
        help='path to the file with the latent representations to reconstruct')
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
### Auxiliary functions
def PC_html(PC1, figname):
    ''' Plot the point clouds as Plotly .html files
    Input:
      - PC: point cloud, type: array, (-1, 3)
      - figname: strig with the name of the output file
    Output:
      - *.html file, saved in the path described in figname
    '''
    ## Assigning the point to the corresponding variables for Plotly
    trace1 = go.Scatter3d(
        x=np.array(PC1[:,0]).flatten(),
        y=np.array(PC1[:,1]).flatten(),
        z=np.array(PC1[:,2]).flatten(),
        mode='markers',
        marker=dict(
            size=7,
            line=dict(
                color='rgb(255, 255, 255)',
                width=0.1
            ),
            opacity=1.0,
            color='rgb(255, 0, 0)',
            colorscale='Viridis'
        )
    )
    data = [trace1,]

    ## Defining the layout of the plot (scatter 3D)
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        autosize=True,
        scene=dict(
            camera=(dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.01, y=2.5, z=0.01)
            )
            ), aspectmode='data'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    
    ## Plot and save figure
    plot(fig, filename=figname, auto_open=False)

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

## Latent representation values
lr_testing = np.array(pd.read_csv(path_to_list, header=None))
lr_testing = np.reshape(lr_testing, (lr_testing.shape[0], -1, 1))

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
n_shapes = lr_testing.shape[0]

# ==============================================================================
### Visualizing point clouds
## Create directory to save the files
dir_pc_plot = str.format("{}/point_cloud_generation", top_dir)
os.system(str.format("mkdir {}", dir_pc_plot))

## Start session
with tf.Session() as sess:
    # import graph data
    new_saver = tf.train.import_meta_graph(TFmetaFile, clear_devices=True)
    new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
    graph = tf.get_default_graph()
    # import network layers
    latent_rep = graph.get_tensor_by_name("latent_rep:0")
    # - Point clouds
    point_clouds = graph.get_tensor_by_name("PC:0")
    if flagVAE: dpout = graph.get_tensor_by_name("do_rate:0")

    ## Latent representation for all input point clouds
    if flagVAE:
        pc_batch = sess.run(point_clouds,\
            feed_dict={latent_rep: lr_testing, dpout: 1.0})
    else:
        pc_batch = sess.run(point_clouds,\
            feed_dict={latent_rep: lr_testing})
    
    ## Plot and save point clouds
    for i in range(n_shapes):
        ## Plot point cloud        
        # - filename
        html_name = \
            "{}/PC_{}_reconstruction.html".format(dir_pc_plot, i)
        # - Plot
        PC_html(pc_batch[i,:,:], html_name)

        ## Save xyz file
        pd.DataFrame(pc_batch[i,:,:]).to_csv(
            "{}/PC_{}_reconstruction.xzy".format(dir_pc_plot, i),
            header=None, index=None, sep=" ")

exit()
#EOF
