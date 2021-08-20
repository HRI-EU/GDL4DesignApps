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

## Scripts for generating the tensorflow graphs corresponding to the 
architecture of the vanilla and variational 3D point cloud autoencoders
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
         Sneha Saha  <sneha.saha@honda-ri.de>
"""

# ==============================================================================
## Import Libraries
# General purpose
import os
import time
import sys

# Mathematical / Scientific tools
import numpy as np
import pandas as pd
import random
import tensorflow as tf

# ==============================================================================
### Assign initial weights/tensors 
## Encoder, for training
def encoder_layer_gen(encoder_layers, pc_size):
    ''' Assigning the initial parameter values to the convolution layers
    Inputs:
      - encoder_layers: array with the number of features for each convolution
      layer, shape = (-1)
      - pc_size: number of points in the point clouds, type: int
    Outputs:
      - x: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
      - conv_layers: dictionary with tensors (filters) for the convolution 
      layers, type: dictionary[<f_conv_layer_0...(n. layers-1)>]
      - bias_conv_layers: dictionary with the bias tensors for the 
      convolution layers, type: dictionary[<b_conv_layer_0...(n. layers-1)>]
    '''

    ## Input point cloud, placeholder
    x = tf.placeholder(tf.float32, (None, pc_size, 3), name="x")
    
    ## Assign convolution layers
    conv_layers = {}
    bias_conv_layers = {}
    for i in range(len(encoder_layers)):
        # First layer: dependent on the size of the point clouds and fixed
        #number of features (3)
        if i == 0:
            # Convolution weights
            conv_layers[str(i)] = tf.Variable(\
                tf.random_normal(\
                [1, 3, encoder_layers[i]], mean=0.0, stddev=0.1, \
                    dtype=tf.float32, seed=0), name=str.format(\
                    "w_conv_layer_{}", i))
            # Bias
            bias_conv_layers[str(i)] = tf.Variable(\
                tf.random_normal(\
                    [encoder_layers[i]],\
                mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                name=str.format("b_conv_layer_{}",i))
        # Further layers
        else:
            # Convolution weights
            conv_layers[str(i)] = tf.Variable(\
                tf.random_normal(\
                [1, encoder_layers[i-1], encoder_layers[i]], \
                    mean=0.0, stddev=0.1, \
                    dtype=tf.float32, seed=0), name=str.format(\
                    "w_conv_layer_{}", i))
            # Bias
            bias_conv_layers[str(i)] = tf.Variable(\
                tf.random_normal(\
                    [encoder_layers[i]],\
                mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                name=str.format("b_conv_layer_{}",i))
    
    return(x, conv_layers, bias_conv_layers)

## (Variational-)Encoder, for testing
def encoder_layer_assign(encoder_layers, pc_size):
    ''' Assigning learned parameter values to the convolution layers
    Inputs:
      - encoder_layers: dictionary with the trained values of the weights
      and bias. Keys: "w_conv_layer_{}" for weights, "b_conv_layer_{}" for bias
    Outputs:
      - x: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
      - conv_layers: dictionary with tensors (filters) for the convolution 
      layers, type: dictionary[<f_conv_layer_0...(n. layers-1)>]
      - bias_conv_layers: dictionary with the bias tensors for the 
      convolution layers, type: dictionary[<b_conv_layer_0...(n. layers-1)>]
    '''

    ## Input point cloud, placeholder
    x = tf.placeholder(tf.float32, (None, pc_size, 3), name="x")

    ## Number of layers in the encoder
    n_layers = int(0.5*len(encoder_layers.keys()))
    
    ## Assign convolution layers
    conv_layers = {}
    bias_conv_layers = {}
    for i in range(n_layers):
        # Convolution weights
        w = encoder_layers["w_conv_layer_{}".format(i)]
        conv_layers[str(i)] = tf.constant(\
            np.reshape(w, (1, w.shape[0], w.shape[1])),
            dtype=tf.float32, name=str.format(\
                    "w_conv_layer_{}", i))
        # Bias
        bias_conv_layers[str(i)] = tf.constant(\
            encoder_layers["b_conv_layer_{}".format(i)].flatten(),
            dtype=tf.float32,\
            name=str.format("b_conv_layer_{}",i))
    
    return(x, conv_layers, bias_conv_layers)

## Variational-Encoder, for training
def vae_encoder_layer_gen(encoder_layers, pc_size):
    ''' Assigning the initial parameter values to the convolution layers
    Inputs:
      - encoder_layers: array with the number of features for each convolution
      layer, shape = (-1)
      - pc_size: number of points in the point clouds, type: int
    Outputs:
      - x: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
      - keep_prob: input placeholder of the keeping probability of the last
      convolution layer dropout
      - conv_layers: dictionary with tensors (filters) for the convolution
      layers, type: dictionary[<f_conv_layer_0...(n. layers-1)>]
      - bias_conv_layers: dictionary with the bias tensors for the
      convolution layers, type: dictionary[<b_conv_layer_0...(n. layers-1)>]
    '''

    ## Input point cloud, placeholder
    x = tf.placeholder(tf.float32, (None, pc_size, 3), name="x")

    ## Drop rate, placeholder
    keep_prob = tf.placeholder(tf.float32, name="do_rate")
    
    ## Assign weights of the convolution layers
    conv_layers = {}
    bias_conv_layers = {}
    for i in range(len(encoder_layers)):
        # First layer: dependent on the size of the point clouds and fixed
        # number of features (3)
        if i == 0:
            # Convolution weights
            conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [1, 3, encoder_layers[i]], mean=0.0, stddev=0.1, \
                    dtype=tf.float32, seed=0), name=str.format( \
                    "w_conv_layer_{}", i))
            # Bias
            bias_conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [encoder_layers[i]], \
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0), \
                name=str.format("b_conv_layer_{}", i))

        # layer that calculates the means of the distributions
        elif i == (len(encoder_layers)-2):
            # Convolution weights
            conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [1, encoder_layers[i - 1], encoder_layers[i]], \
                    mean=0.0, stddev=0.1, \
                    dtype=tf.float32, seed=0), name=str.format( \
                    "w_conv_layer_{}", i))
            # Bias
            bias_conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [encoder_layers[i]], \
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0), \
                name=str.format("b_conv_layer_{}", i))

        # layer that calculates the std. dev. of the distributions
        elif i == (len(encoder_layers)-1):
            # Convolution weights
            conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [1, encoder_layers[i - 2], encoder_layers[i]], \
                    mean=0.0, stddev=0.1, \
                    dtype=tf.float32, seed=0), name=str.format( \
                    "w_conv_layer_{}", i))
            # Bias
            bias_conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [encoder_layers[i]], \
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0), \
                name=str.format("b_conv_layer_{}", i))
        
        # Further layers
        else:
            # Convolution weights
            conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [1, encoder_layers[i - 1], encoder_layers[i]], \
                    mean=0.0, stddev=0.1, \
                    dtype=tf.float32, seed=0), name=str.format( \
                    "w_conv_layer_{}", i))
            # Bias
            bias_conv_layers[str(i)] = tf.Variable( \
                tf.random_normal( \
                    [encoder_layers[i]], \
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0), \
                name=str.format("b_conv_layer_{}", i))

    return (x, keep_prob, conv_layers, bias_conv_layers)

## Decoder, for training
def decoder_layer_gen(latent_layer, decoder_layers):
    ''' Assigning the initial parameter values to the *fully* connected layers
    Inputs:
      - latent_layer: number of latent variables, type: int
      - decoder_layers: array with the number of hidden neurons for each *fully*
      connected layer, type: array, (-1)
    Outputs:
      - fc_layers: dictionary with tensors containing the initial weights of the
      *fully* connected layers, type: dictionary[<fc_conv_layer_0...(n. layers)
      >]
      - bias_fc_layers: dictionary with the bias tensors for the 
      *fully* connected layers, type: dictionary[<b_fc_layer_0...(n. layers-1)
      >]
    '''
    fc_layers = {}
    bias_fc_layers = {}
    ## Decoder: Fully Connected Layers
    # First fully connected layer
    fc_layers[str(0)] = tf.Variable(tf.random_normal(\
        [decoder_layers[0],latent_layer], \
            mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
            name=str.format("w_fc_layer_{}", 0))
    bias_fc_layers[str(0)] = tf.Variable(tf.random_normal(\
        [decoder_layers[0], 3],\
        mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
        name=str.format("b_fc_layer_{}",0))
    # Further layers
    for i in range(1, len(decoder_layers)):
        fc_layers[str(i)] = tf.Variable(tf.random_normal(\
            [decoder_layers[i], decoder_layers[i-1]], \
                mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                name=str.format("w_fc_layer_{}", i))
        bias_fc_layers[str(i)] = tf.Variable(tf.random_normal(\
            [decoder_layers[i],\
             3], mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
             name=str.format("b_fc_layer_{}",i))
    
    return(fc_layers, bias_fc_layers)

## (Variational-)Decoder, for testing
def decoder_layer_assign(decoder_layers):
    ''' Assigning learned parameter values to the *fully* connected layers
    Inputs:
      - decoder_layers: dictionary with the trained weights of the decoder.
      The keys of the dictionary should be the name of the tensors.
    Outputs:
      - fc_layers: dictionary with tensors containing the initial weights of the
      *fully* connected layers, type: dictionary[<fc_conv_layer_0...(n. layers)
      >]
      - bias_fc_layers: dictionary with the bias tensors for the 
      *fully* connected layers, type: dictionary[<b_fc_layer_0...(n. layers-1)
      >]
    '''
    fc_layers = {}
    bias_fc_layers = {}

    ## Number of layers
    n_layers = int(0.5*len(decoder_layers.keys()))

    ## Decoder: Fully Connected Layers
    for i in range(n_layers):
        fc_layers[str(i)] = tf.constant(\
            decoder_layers["w_fc_layer_{}".format(i)], dtype=tf.float32,\
                name=str.format("w_fc_layer_{}", 0))
        bias_fc_layers[str(i)] = tf.constant(\
            decoder_layers["b_fc_layer_{}".format(i)], dtype=tf.float32,\
            name=str.format("b_fc_layer_{}",0))
    
    return(fc_layers, bias_fc_layers)

# ==============================================================================
### Generate the Graph
## Encoder
def ENCODER(x, conv_layers, bias_conv_layers, latent_layer):
    ''' ENCODER: defines the graph that represents the encoder
    Inputs:
      - x: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
      - conv_layers: dictionary with tensors (filters) for the convolution 
      layers, type: dictionary[<f_conv_layer_0...(n. layers-1)>]
      - bias_conv_layers: dictionary with the bias tensors for the 
      convolution layers, type: dictionary[<b_conv_layer_0...(n. layers-1)>]
      - latent_layer: number of latent variables, type: int
    Outputs:
      - latent_rep: tensor with the latent variables corresponding to the input
      x, type: tensor, (-1, latent_layer, 1)
    '''

    # Initialize empty dictionaries for assigning the layers
    conv_op = {}
    actv_conv_op = {}
    # Assign initial (n-1) convolution layers
    for i in range(len(conv_layers)-1):
        # First layer, dependent on the dimensionality of the input
        #point clouds
        if i == 0:
            conv_op[str(i)] = tf.nn.conv1d(\
                x, conv_layers[str(i)], stride=(1), padding="SAME") \
                    + bias_conv_layers[str(i)]
            actv_conv_op[str(0)] = tf.nn.relu(conv_op[str(i)], name="CLayer_0")
            #math.tan(conv_op[str(i)], name="CLayer_0")
            #nn.relu(conv_op[str(i)], name="CLayer_0")
        # Further intermediate layers
        else:
            conv_op[str(i)] = tf.nn.conv1d(\
                actv_conv_op[str(i-1)], conv_layers[str(i)], \
                    stride=(1), padding="SAME") \
                        + bias_conv_layers[str(i)]
            actv_conv_op[str(i)] = tf.nn.relu(conv_op[str(i)],\
                name=str.format("CLayer_{}", i))
            
    # Last convolution layer
    i = len(conv_layers)-1
    conv_op[i] = tf.nn.conv1d(\
        actv_conv_op[str(i-1)], conv_layers[str(i)],\
            stride=(1), padding="SAME") + bias_conv_layers[str(i)]
    actv_conv_op[str(i)] = tf.nn.tanh(conv_op[i], \
        name=str.format("CLayer_{}", i))

    # Max pooling operation
    max_pool = tf.reduce_max(actv_conv_op[str(i)], axis=1)
    
    # Extracting the latent representations
    latent_rep = tf.reshape(max_pool, shape=(-1, latent_layer, 1),\
        name="latent_rep")
    return latent_rep

## Variational-Encoder
def vae_ENCODER(x, conv_layers, bias_conv_layers, latent_layer, keep_prob):
    ''' ENCODER: defines the graph that represents the encoder
    Inputs:
      - x: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
      - conv_layers: dictionary with tensors (filters) for the convolution
      layers, type: dictionary[<f_conv_layer_0...(n. layers-1)>]
      - bias_conv_layers: dictionary with the bias tensors for the
      convolution layers, type: dictionary[<b_conv_layer_0...(n. layers-1)>]
      - latent_layer: number of latent variables, type: int
      - keep_prob: tensor with (1-drop_out ratio)
    Outputs:
      - latent_rep: tensor with the latent variables corresponding to the input
      x, type: tensor, (-1, latent_layer, 1)
      - mu_encoder: 
      - logvar_encoder: 
    '''

    # Initialize empty dictionaries for assigning the layers
    conv_op = {}
    conv_op_norm = {}
    actv_conv_op = {}
    
    # Assign initial (n-1) convolution layers
    for i in range(len(conv_layers) - 2):
        # First layer, dependent on the dimensionality of the input
        # point clouds
        if i == 0:
            # Convolution
            conv_op[str(i)] = tf.nn.conv1d( \
                x, conv_layers[str(i)], stride=(1), padding="SAME") \
                              + bias_conv_layers[str(i)]
            # Batch normalization
            conv_op_norm[str(0)] = tf.layers.batch_normalization(\
                    conv_op[str(i)], momentum=0.9, training=True)
            # Leaky ReLU
            actv_conv_op[str(i)] = tf.nn.leaky_relu(\
                conv_op_norm[str(0)], name="CLayer_0")

        # Further intermediate layers
        else:
            conv_op[str(i)] = tf.nn.conv1d( \
                actv_conv_op[str(i - 1)], conv_layers[str(i)], \
                stride=(1), padding="SAME") \
                              + bias_conv_layers[str(i)]
            # Batch normalization
            conv_op_norm[str(i)] = tf.layers.batch_normalization(\
                conv_op[str(i)], momentum=0.9, training=True)
            # Leaky ReLU
            actv_conv_op[str(i)] = tf.nn.leaky_relu(\
                conv_op_norm[str(i)], name=str.format("CLayer_{}", i))

    # Max pooling layer
    max_pool = tf.reshape(\
        tf.reduce_max(actv_conv_op[str(i)], axis=1), shape=(-1, 1, 256))

    # Mu (mean) layer
    i = len(conv_layers) - 2
    conv_op[str(i)] = tf.nn.dropout(\
        tf.nn.conv1d(\
            max_pool, conv_layers[str(i)], stride=(1), padding="SAME")\
        + bias_conv_layers[str(i)], keep_prob=keep_prob)
    actv_conv_op[str(i)] = tf.identity(\
        conv_op[str(i)], name=str.format("CLayer_{}", i))
    # Tensor for mean values
    mu_encoder = actv_conv_op[str(i)]

    # Sigma (std. dev) layer
    i = len(conv_layers) - 1
    conv_op[str(i)] = tf.nn.conv1d( \
        max_pool, conv_layers[str(i)], stride=(1), padding="SAME")\
            + bias_conv_layers[str(i)]
    actv_conv_op[str(i)] = tf.nn.sigmoid(conv_op[str(i)], \
                                      name=str.format("CLayer_{}", i))
    # Tensor with logvar values
    logvar_encoder = actv_conv_op[str(i)]

    # re-parameterization trick
    # <name of the variable>
    eps = tf.random.normal(
        shape=tf.shape(logvar_encoder),
        mean=0, stddev=1, dtype=tf.float32, seed=0,\
        name=str.format("random_eps_{}", i))
    # <name of the variable>
    std_encoder = tf.exp(logvar_encoder * 0.5)
    # Sample latent variables
    z = mu_encoder + tf.multiply(std_encoder, eps)

    # Extracting the latent representations
    latent_rep = tf.reshape(z, shape=(-1, latent_layer, 1), \
                            name="latent_rep")
    return (latent_rep, mu_encoder, logvar_encoder)

## Decoder
def DECODER(latent_vec, decoder_layers, fc_layers, bias_fc_layers):
    ''' DECODER: defines the graph that represents the decoder
    Inputs:
      - latent_vec: tensor with the latent representations corresponding to 
      the input x, type: tensor, (-1, latent_layer, 1)
      - decoder_layers: array with the number of hidden neurons for each *fully*
      connected layer, type: array, (-1)
      - fc_layers: dictionary with tensors containing the initial weights of the
      *fully* connected layers, type: dictionary[<fc_conv_layer_0...(n. layers)
      >]
      - bias_fc_layers: dictionary with the bias tensors for the 
      *fully* connected layers, type: dictionary[<b_fc_layer_0...(n. layers-1)
      >]
    Outputs:
      - PC: tensor with the retrieved point clouds corresponding to the input
      x, type: tensor, (-1, pc_size, 3)
    '''
    fc_op = {}
    # Adapt code for 3 Dimensions
    latent_rep_trip = tf.concat((latent_vec, latent_vec, latent_vec), 2)

    # Fully Connected Layers
    for i in range(len(decoder_layers)-1):
        # First layer after the latent representation
        if i == 0:
            fc_op[str(i)] = tf.nn.relu(tf.scan(\
                lambda a, z: tf.add(tf.matmul(fc_layers[str(i)], z), \
                    bias_fc_layers[str(i)]), latent_rep_trip,\
                        initializer=tf.zeros((decoder_layers[i], 3))),\
                            name=str.format("FCLayer{}", 0))
        # Intermediate layers, before the output
        else:
            fc_op[str(i)] = tf.nn.relu(tf.scan(\
                lambda a, z: tf.add(tf.matmul(fc_layers[str(i)], z), \
                    bias_fc_layers[str(i)]), fc_op[str(i-1)],\
                        initializer=tf.zeros((decoder_layers[i], 3))),\
                            name=str.format("FCLayer{}", i))
    # Last layer, with the same shape as the input tensor x
    i = len(decoder_layers)-1
    fc_op[str(i)] = tf.nn.sigmoid(tf.scan(\
        lambda a, z: tf.add(tf.matmul(fc_layers[str(i)], z), \
            bias_fc_layers[str(i)]), fc_op[str(i-1)],\
                initializer=tf.zeros((decoder_layers[i], 3))))

    # Output tensor: retrieved point clouds
    PC = tf.reshape(fc_op[str(i)], shape=(-1, decoder_layers[i], 3), name="PC")
    return PC

# Variational-decoder
def vae_DECODER(latent_vec, decoder_layers, fc_layers, bias_fc_layers):
    ''' DECODER: defines the graph that represents the decoder
    Inputs:
      - latent_vec: tensor with the latent representations corresponding to
      the input x, type: tensor, (-1, latent_layer, 1)
      - decoder_layers: array with the number of hidden neurons for each 
      *fully* connected layer, type: array, (-1)
      - fc_layers: dictionary with tensors containing the initial weights of 
      the *fully* connected layers, 
      type: dictionary[<fc_conv_layer_0...(n. layers)>]
      - bias_fc_layers: dictionary with the bias tensors for the
      fully connected layers, type: dictionary[<b_fc_layer_0...(n. layers-1)>]
    Outputs:
      - PC: tensor with the retrieved point clouds corresponding to the input
      x, type: tensor, (-1, pc_size, 3)
    '''
    fc_op = {}
    fc_op_norm = {}
    # Adapt code for 3 Dimensions
    latent_rep_trip = tf.concat((latent_vec, latent_vec, latent_vec), 2)

    # Fully Connected Layers
    for i in range(len(decoder_layers) - 1):
        # First layer after the latent representation
        if i == 0:
            fc_op[str(i)] = tf.nn.leaky_relu(\
                tf.scan(\
                    lambda a, z: tf.layers.batch_normalization(\
                        tf.add(\
                            tf.matmul(\
                                fc_layers[str(i)], z\
                            ), bias_fc_layers[str(i)]\
                        ), momentum=0.9, training=True\
                    ), latent_vec, initializer=tf.zeros(\
                        (decoder_layers[i], 3))\
                ), name=str.format("FCLayer{}", 0))

        # Intermediate layers, before the output
        else:
            fc_op[str(i)] = tf.nn.leaky_relu(\
                tf.scan(\
                    lambda a, z: tf.layers.batch_normalization(\
                        tf.add(\
                            tf.matmul(\
                                fc_layers[str(i)], z\
                            ), bias_fc_layers[str(i)]\
                        ), momentum=0.9, training=True\
                    ), fc_op[str(i - 1)], initializer=tf.zeros(\
                        (decoder_layers[i], 3))\
                ), name=str.format("FCLayer{}", i))

    # Last layer, with the same shape as the input tensor x
    i = len(decoder_layers) - 1
    fc_op[str(i)] = tf.nn.sigmoid(tf.scan( \
        lambda a, z: tf.add(tf.matmul(fc_layers[str(i)], z), \
                            bias_fc_layers[str(i)]), fc_op[str(i - 1)], \
        initializer=tf.zeros((decoder_layers[i], 3))))

    # Output tensor: retrieved point clouds
    PC = tf.reshape(fc_op[str(i)], shape=(-1, decoder_layers[i], 3), name="PC")
    return PC

# EOF
