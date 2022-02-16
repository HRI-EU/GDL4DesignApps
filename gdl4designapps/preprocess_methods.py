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


Script for shrink-wrapping 3D point clouds based on an initial polygonal 
mesh

Shrink-wrapping algorithm based on  L. P. Kobbelt, J. Vorsatz, U. Labsik, 
and H. P. Seidel, “A Shrink Wrapping Approach to rRmeshing Polygonal 
Surfaces” Computer Graphics Forum, vol. 18, no. 3, pp. 119–130, 1999.

Pre-requisites:
 - Python      3.6.10
 - numpy       1.19.1
 - Trimesh     3.8.10
 - Scipy       1.3.1
 - Ubuntu      18.04

Copyright (c)
Honda Research Institute Europe GmbH
Carl-Legien-Str. 30
63073 Offenbach/Main
Germany

UNPUBLISHED PROPRIETARY MATERIAL.
ALL RIGHTS RESERVED.

Authors: Thiago Rios, Sneha Saha
Contact: gdl4designapps@honda-ri.de
"""

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------
# Basic tools
from random import shuffle
import numpy as np
import os
import time
import random

# Geometric data tools
from scipy.spatial.distance import cdist
import trimesh

# Visualization tools
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# Custom colormap
cdict = {'red':   [[0.0,   0/256,    0/256],
                   [0.50,  149/256,  149/256],
                   [1.0,   200/256,  200/256]],
        
         'green': [[0.0,   51/256,   51/256],
                   [0.50,  157/256,  157/256],
                   [1.0,   16/256,   16/256]],
        
         'blue':  [[0.0,   76/256,   76/256],
                   [0.50,  168/256,  168/256],
                   [1.0,   46/256,   46/256]]}
cm_custm = matplotlib.colors.LinearSegmentedColormap('cstmmap', cdict, N=256)
import matplotlib.pyplot as plt
import seaborn as sns
palette = ["#00334c", "#959da8", "#c8102e"]

# Data science/machine learning tools
import pandas as pd
import tensorflow as tf

# Other tools
import scipy.sparse as sprs


## Initialize
np.random.seed(0)
random.seed(0)
# ------------------------------------------------------------------------------
# METHODS
# ------------------------------------------------------------------------------

# Methods for sampling CAE models into point clouds
class CAE2PC:
    # Load obj scenes/meshes as meshes
    def as_mesh(obj_file):
        ''' Function to convert objects from scene to mesh

        Input:
          - obj_file: path to the file (obj) to be loaded
        
        Output:
          - mesh: triangular mesh of the geometry in the file
        '''
        # Load geometry from file
        geom = trimesh.load(obj_file)
    
        # In case the object is a scene
        try:
            if isinstance(geom, trimesh.Scene):
                # Check if the scene contains objects
                if len(geom.geometry) == 0:
                    mesh = None
                else:
                    # Remove scene data
                    mesh = trimesh.util.concatenate(
                        tuple(trimesh.Trimesh(vertices=g.vertices, 
                                              faces=g.faces)
                            for g in geom.geometry.values()))
            
            # In case the file only contains a mesh
            else:
                assert(isinstance(geom, trimesh.Trimesh))
                mesh = geom
        except:
            mesh = geom    
        return(mesh)
    
    # Shrink-warpping algorihtm
    def swm(mesh_0, pc_targ, alpha=0.5, n_iter=10, d_thresh=0, smooth_iter=2):
        ''' Shrink wrapping meshing algorithm
        
        Input parameters:
          - mesh_0:  Initial mesh, loaded uS_ing trimesh 
          (trimesh.load("mesh_path.stl"))
          - pc_targ: Target shape represented as a point cloud (numpy array, (N,3))
          - alpha: Step size for shrinkigng the mesh. The lower the finer the
          modifications at each shrinking step.
          - n_iter: Maximum number of shrinking iterations
          - dist_thresh: Threshold for the Hausdorff distance (HDD). If the HDD 
          becomes lower than the threshold, the algorithm stops.
          - smooth_iter: Number of smoothing iterations uS_ing the laplacian filter.
        
        Output parameters:
          - mesh_smooth: Smoothed shrink-wrapped mesh (trimesh compatible)
          - mesh_smooth.vertices: output point cloud with nodes ordered according 
          to the topology of the initial mesh.
          - hdd: Hausdorff distance between the shrink-wrapped mesh and the target 
          point cloud.
        '''
        
        ## Generate base mesh
        # Vertices from the initial mesh
        pc_0 = np.array(mesh_0.vertices)
    
        # Find center and main dimensions of the shape
        pc_targ_center = 0.5*(np.min(pc_targ, axis=0) + np.max(pc_targ, axis=0))
        pc_targ_maindm = (np.max(pc_targ, axis=0) - np.min(pc_targ, axis=0))
    
        # Adjust initial mesh
        pc_0 = pc_0*(1.1*pc_targ_maindm/(np.max(pc_0, axis=0) - \
               np.min(pc_0, axis=0)))
        pc_0 -= -pc_targ_center + 0.5*(np.min(pc_0, axis=0) + \
                np.max(pc_0, axis=0))
    
        ## Shrinking
        pc_iter = np.array(pc_0)
        mesh_iter = mesh_0.copy()
        for i in range(n_iter):
            # update points
            id_points = np.argmin(cdist(pc_iter, pc_targ, "euclidean"), axis=1)
            pc_iter += alpha*(pc_targ[id_points,:] - pc_iter)
            mesh_iter.vertices = pc_iter
            
            # Check convergence
            max_dist = np.max(np.min(cdist(pc_iter, pc_targ, "euclidean"), axis=1))
            if max_dist < d_thresh: break
        mesh_iter.vertices = pc_iter
    
        ## Smoothing
        if smooth_iter > 0:
            mesh_smooth = trimesh.smoothing.filter_laplacian(
                mesh_iter, lamb=0.5, iterations=smooth_iter
            )
        else:
            mesh_smooth = mesh_iter
        
        ## Hausdorff distance
        hdd = 0.5*(np.mean(np.min(cdist(mesh_smooth.vertices, pc_targ, "euclidean"), axis=0))\
            + np.mean(np.min(cdist(mesh_smooth.vertices, pc_targ, "euclidean"), axis=1)))
        
        return(mesh_smooth, mesh_smooth.vertices, hdd)

    # Shrink-wrapping a data set of CAE meshes
    def swm_data_set(data_set_path, ref_mesh_path, out_dir, swm_config=None, 
                    stl=False):
        ''' Algorithm to sample 3D point clouds uS_ing shrink-wrapping meshing
        Readable CAE files: .OBJ, .STL, .XYZ, .PLY

        Input parameters:
          - data_set_path: path to the directory with shapes to be sampled
          - ref_mesh_path: path to the reference mesh, which will be shrunk
          - out_dir: prefix of the output directory name
          - swm_config: dictionary with the configuration to perform the
          shrink-wrapping sampling {'shrink': shrinking steps, 'alpha': alpha, 
          'smooth': smoothing steps}
          - stl: option to output the shrink-wrapped stl meshes
        
        Output parameters: None
          - The script saves the files directly to the specified directories.
        '''
    
        ## Check data set path
        if not os.path.exists(data_set_path):
            print("ERROR! Path to the data set does not exist")
            print("Path: {}".format(data_set_path))
            return()
        
        ## Check if reference mesh exists
        if not os.path.exists(ref_mesh_path):
            print("ERROR! Path to the reference mesh does not exist")
            print("Path: {}".format(ref_mesh_path))
            return()
        else:
            mesh_ref = CAE2PC.as_mesh(trimesh.load(ref_mesh_path))
    
        ## Check if output directory exists
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        # Point clouds
        out_xyz_dir = "{}/xyz".format(out_dir)
        if not os.path.exists(out_xyz_dir): os.mkdir(out_xyz_dir)
        # Meshes
        if stl:
            out_stl_dir = "{}/stl".format(out_dir)
            if not os.path.exists(out_stl_dir): os.mkdir(out_stl_dir)
    
        ## Shrink-wrapping configuration
        std_config = {'shrink': 6, 'alpha': 0.5, 'smooth': 2}
        for k in std_config.keys():
            try:
                swm_config[k]
            except:
                try:
                    swm_config[k] = std_config[k]
                except:
                    swm_config = {}
                    swm_config[k] = std_config[k]
        
        ## Run the shrink-wrapping
        # List of shapes
        shape_list = os.listdir(data_set_path)
        # Flag to generate log files
        flag_log = False
        # Load and sample
        for s in shape_list:
            ## Load shape
            # In case the shape is a mesh (OBJ or STL)
            try:
                s_target = CAE2PC.as_mesh(
                                trimesh.load("{}/{}".format(data_set_path,s))
                                ).vertices
            # In case the shape is a point cloud (XYZ, CSV or DAT)
            except:
                if s[-3:] == "xyz":
                    sep_file=" "
                else:
                    sep_file=","
                try:
                    s_target = np.array(pd.read_csv("{}/{}".
                                                format(data_set_path,s),
                                        header=None, sep=sep_file))[:,:3]
                except:
                    continue
            
            ## Shrink-wrapping        
            swm_mesh, swm_pcld, hdd = CAE2PC.swm(mesh_ref, s_target,\
                alpha=swm_config['alpha'], n_iter=swm_config['shrink'],\
                    dist_thresh=0, smooth_iter=swm_config['smooth'])
            # Export files
            # - Point cloud
            pd.DataFrame(swm_pcld).to_csv("{}/{}".format(\
                                          out_xyz_dir, s[:-3]+'xyz'),\
                                          header=None, index=None, sep=" ")
            # - STL file
            if stl:
                swm_mesh.export("{}/{}".format(out_stl_dir, s[:-3]+'stl'))
            # Log Hausdorff distance
            if os.path.exists("{}/log_hdd.dat".format(out_dir)):
                # Remove potential old log files and create new one
                if not flag_log:
                    os.system("rm {}/log_hdd.dat".format(out_dir))
                    flag_log = True
                    file_log = open("{}/log_hdd.dat".format(out_dir),"w")
                    file_log.write("{}, {}\n".format(s, hdd))
                    file_log.close()
                else:
                    # Append information to the log file
                    file_log = open("{}/log_hdd.dat".format(out_dir),"a")
                    file_log.write("{}, {}\n".format(s, hdd))
                    file_log.close()
            else:
                # Create a new log file, if no older file exists
                file_log = open("{}/log_hdd.dat".format(out_dir),"w")
                file_log.write("{}, {}\n".format(s, hdd))
                file_log.close()
            
        return()
    
    # High-pass filter
    def hpf(mesh):
        ''' High-pass filtering for meshes
        The function calculates the sampling probability of the mesh nodes
        based on the proximity to high frequency transitions (e.g. edges).

        Input:
          - mesh: mesh object loaded with the library trimesh

        Output:
          - probsamp: sampling probability for each mesh vertex
        '''
    
        ## Calculate the Adjacency matrix
        # Load mesh nodes
        vertices = mesh.vertices
        # Load mesh edges
        edges = mesh.edges
        # Generate sparse adjacency and degree matrices
        rw_ind = [] # row indices
        cl_ind = [] # column indices
        data_A = [] # data of the adjacency matrix
        data_D = np.zeros(vertices.shape[0]) # data of the degree matrix
        # Assign indices and data according to the mesh edges 
        for i in range(edges.shape[0]):
            rw_ind += [edges[i,0], edges[i,1]]
            cl_ind += [edges[i,1], edges[i,0]]
            data_A += [1, 1]
            data_D[edges[i,0]] += 1
            data_D[edges[i,1]] += 1
        # Assign data to sparse adjacency matrix (A)
        A = sprs.coo_matrix(
                                 (data_A, (rw_ind, cl_ind)), 
                                 shape=(vertices.shape[0], vertices.shape[0])
                                 ).tocsc()
        # Assign data to sparse degree matrix (D)
        D = sprs.spdiags(data_D, diags=0, m=vertices.shape[0], 
                              n=vertices.shape[0]).tocsc()
        
        # Calculate the transition matrix T (Filter)
        T = np.matmul(sprs.linalg.inv(D).toarray(), A.toarray())
        # Calculate the filter response at each node
        x_rs = np.zeros(vertices.shape)
        for i in range(vertices.shape[0]):
            x_rs[i,:] = vertices[i,:] - np.matmul(T[i,0:],vertices)
        
        # Feature extraction operator
        probsamp = np.linalg.norm(x_rs, axis=1)**2
        # Calculate sampling probability
        probsamp = probsamp/np.sum(probsamp)
        return(probsamp)

    # Function to calculate the HPF sampling probability for a CAE data set
    def hpf_data_set(data_set_path, out_dir):
        ''' Algorithm to calculate the vertex sampling probability using 
        a graph high-pass filter. Readable CAE files: .OBJ, .STL, .XYZ

        Input parameters:
          - data_set_path: path to the directory with shapes to be sampled
          - out_dir: prefix of the output directory name

        Output parameters: None
          - The script saves the files directly to the specified directories.
        '''
    
        ## Check data set path
        if not os.path.exists(data_set_path):
            print("ERROR! Path to the data set does not exist")
            print("Path: {}".format(data_set_path))
            return()
        
        ## Check if output directory exists
        if not os.path.exists(out_dir): os.mkdir(out_dir)
    
        ## Run the shrink-wrapping
        # List of shapes
        shape_list = os.listdir(data_set_path)
        # Load and sample
        for s in shape_list:
            try:
                # Load shape
                mesh = CAE2PC.as_mesh(
                                trimesh.load("{}/{}".format(data_set_path,s))
                                )
                # Calculate sampling probability
                samp_prob = CAE2PC.hpf(mesh)
                # Save file
                pd.DataFrame(samp_prob).to_csv("{}/{}".format(out_dir, s[:-3]+"dat",
                                               ), header=None, index=None)
            except:
                continue
        return()

    # Algorithm for sampling 3D point clouds for deep-generative tasks
    #(training and applications)
    def pc_sampling(data_set_path, n_points, n_pclouds=None, p_samp=None):
        ''' Point Cloud sampling algorithm

        Input:
          - dataset_path: path to the directory containing point clouds. 
          Type: str
          - n_points: size of the point cloud. Type: int
          - n_pclouds: number of point clouds to load. Three types of input are
          possible:
              * Not specified (None): all point cloud files are loaded
              * List with file names (list, str): only the specified files are
              loaded
              * Number of files to load (int): the script reads, samples and 
              assigns the first n_pclouds files to the data set.
          - p_samp: sampling probability for each point and point cloud in the
          data set. Type: dictionary, with the name of the files as keys

        Output:
          - dataset_full: resampled batch, type: array, (batch_size, pc_size, 3)
          - log_names: array with the name of the loaded files
        '''
    
        ## Check data set path
        if not os.path.exists(data_set_path):
            print("ERROR! Path to the data set does not exist")
            print("Path: {}".format(data_set_path))
            return([0,], ["FLAG ERROR",])
        else:
            # Get the name of the files/directories in the provided path
            dir_names = os.listdir(data_set_path)
            
        # Check if the number of required point clouds is less than the
        #number of files in the specified path
        # -- Number of files not specified
        if n_pclouds == None:
            n_pclouds = len(dir_names)
        else:
            # -- A number of files was defined as int
            if isinstance(n_pclouds, int):
                if n_pclouds > len(dir_names):
                    print("*** WARNING: user required {} point clouds, \
                        but target directory contains only {} files. \
                        n_pclouds reduced to {}.".format(n_pclouds, 
                        len(dir_names), len(dir_names)))
                    n_pclouds = len(dir_names)
            # -- The target files were specified in a list
            else:
                if isinstance(n_pclouds, list):
                    dir_names = n_pclouds
                    n_pclouds = len(dir_names)
                else:
                    print("** ERROR! Script interrupted.")
                    print("Input 'n_pclouds' not understood. \
                        Please check if the variable was assigned \
                        as None, int or list.")
                    return([0,], ["FLAG ERROR",])
        
        # Iteration over the files/directories contained in the path
        # - Counter
        cntr = 0

        # Pre-allocate the array to assign the sampled point clouds
        dataset_full = np.zeros((n_pclouds, n_points, 3))
        
        # List of shape names (pre-allocated)
        log_names = list(np.array(np.zeros(n_pclouds), dtype='U70'))
    
        # Loop over the files
        print("## Sampling geometries...")
        for i in range(n_pclouds):
            print("File {} of {}".format(i+1, n_pclouds), end='\r')
            # Path to shape i
            shape_name = str.format("{}/{}", data_set_path, dir_names[i])
            # File type
            extension = shape_name[len(shape_name)-3:None]
            # Check if file extension can be read by the script
            ext_allow = ["obj", "ply", "stl", "xyz", "csv"]
            if not os.path.isfile(shape_name) or not extension in ext_allow:
                continue
            
            # Sample according to the type of file
            # STL or OBJ
            if extension in ["obj", "ply", "stl"]:#=="stl" or extension=="obj":
                pc_temp = CAE2PC.as_mesh(trimesh.load_mesh(shape_name)).vertices
                log_names[cntr] = shape_name
            # CSV
            if extension=="csv":
                pc_temp = np.array(\
                            pd.read_csv(shape_name, header=None))
                log_names[cntr] = shape_name
            # XYZ
            if extension=="xyz":
                pc_temp = np.array(\
                            pd.read_csv(shape_name, header=None, \
                                delimiter=" "))[:,0:3]
                log_names[cntr] = shape_name
                
            # Sample point cloud
            if pc_temp.shape[0] == n_points: 
                # If the size is the same as number of samples, 
                # the point cloud is directly assigned to the 
                # data set tensor
                dataset_full[i,:,:] = pc_temp
                # Update counter
                cntr = cntr + 1
            else:
                # For point clouds with different sizes
                # If the point cloud has less samples than
                # required, enable sampling points more than once
                if pc_temp.shape[0] < n_points:
                    replace_opt = True
                else:
                    # If the point cloud has more samples than 
                    # required, each point can be sampled only once
                    replace_opt = False
                # Define the same sampling probability for all the points
                # - In case the probability is specified
                if not p_samp == None:
                    if 1==1:
                        prob_values = np.array(
                                         pd.read_csv("{}/{}".format(
                                            p_samp, dir_names[i][:-3]+"dat")
                                            , header=None))
                        #prob_values = p_samp[dir_names[i]]
                        prob_values = (prob_values+1e-9) / np.linalg.norm(
                                                            prob_values+1e-9,1)
                    else:
                        continue
                # - If not specified, a uniform probability is assigned to 
                #the points
                else:
                    prob_values = np.ones(pc_temp.shape[0])
                    prob_values = prob_values / np.linalg.norm(prob_values,1)
                # Selecting points
                sample_points = np.random.choice(\
                    list(range(pc_temp.shape[0])),\
                    (n_points), p=prob_values.flatten(), \
                    replace=replace_opt)
                # Assigning to the dataset
                dataset_full[cntr,:,:] = np.reshape(pc_temp[sample_points,:],\
                    (1, -1, 3))
                # Update counter
                cntr = cntr + 1
        print(cntr)
        print("\n## Finished sampling! ## \n")
        return(dataset_full[0:cntr,:,:], log_names[0:cntr])

    # Algorithm to normalize 3D point cloud data
    def data_set_norm(pc_batch, out_lim, inp_lim=None):
        ''' Data set normalization
        Algorithm to scale the coordinates of the point clouds to an 
        interval of interest (e.g. [0,1]^3). The aspect ratio of the 
        geometries is preserved.
        Input:
          - data_set: Batch of point clouds. Type: array, (-1, pc_size, 3)
          - out_lim: Output interval. Type: array, (2,)
          - inp_lim: Input interval. If not specified, the range of the data is
          considered instead. Type: array, (2,)
        Output:
          - pc_norm: Normalized data set. Type: array (-1, pc_size, 3)
          - inp_lim: data interval of the original data set. Type: array, (2,)
        '''
    
        # Assign input limits
        if np.array(inp_lim).any()==None:
            inp_lim = np.array([np.min(pc_batch), np.max(pc_batch)])
    
        ## Normalize the data
        # Scale data to [0,1]
        pc_norm = (pc_batch - np.min(inp_lim))/(np.max(inp_lim)-np.min(inp_lim))
        # Scale data to [min, max] of th output limits
        pc_norm = pc_norm*(np.max(out_lim)-np.min(out_lim)) + np.min(out_lim)
    
        return(pc_norm, inp_lim)

# 3D Point Cloud Autoencoder (architecture)
class PC_AE:
    # PC-AE encoder parameters
    def pcae_enc_params(encoder_sizes, pc_size):
        ''' Initial values of the PC-AE encoder parameters

        Input:
          - encoder_sizes: Array with the number of features for each 
          convolution layer. Type <array (-1)>
          - pc_size: Number of points in the point clouds. Type: int

        Output:
          - S_in: Input placeholder of the PC-AE. Type <tensor (-1,pc_size,3)>
          - enc_w: Dictionary with tensors (filters) for the convolution layers.
          Type: <dictionary[enc_w_layer_{},...,enc_w_layer_{}]>
          - enc_b: Dictionary with the bias tensors for the convolution layers. 
          Type: <dictionary[enc_b_layer_{},...,enc_b_layer_{}]>
        '''
    
        ## Input point cloud, placeholder
        S_in = tf.placeholder(tf.float32, (None, pc_size, 3), name="S_in")
        
        ## Assign convolution layers
        enc_w = {}
        enc_b = {}
        for i in range(len(encoder_sizes)):
            # First layer: dependent on the size of the point clouds and fixed
            #number of features (3)
            if i == 0:
                # Convolution weights
                enc_w[i] = tf.Variable(\
                                     tf.random_normal(
                                                    [1, 3, encoder_sizes[i]],
                                                     mean=0.0, stddev=0.1,
                                                     dtype=tf.float32, seed=0),
                                     name=str.format("enc_w_layer_{}", i))
                # Bias
                enc_b[i] = tf.Variable(\
                                     tf.random_normal(
                                                    [encoder_sizes[i]],
                                                     mean=0.0, stddev=0.1,
                                                     dtype=tf.float32, seed=0),
                                     name=str.format("enc_b_layer_{}",i))
            # Further layers
            else:
                # Convolution weights
                enc_w[i] = tf.Variable(\
                                     tf.random_normal(
                                                    [1, encoder_sizes[i-1],
                                                     encoder_sizes[i]],
                                                     mean=0.0, stddev=0.1,
                                                     dtype=tf.float32, seed=0),
                                     name=str.format("enc_w_layer_{}", i))
                # Bias
                enc_b[i] = tf.Variable(\
                                     tf.random_normal(
                                                    [encoder_sizes[i]],
                                                     mean=0.0, stddev=0.1,
                                                     dtype=tf.float32, seed=0),
                                     name=str.format("enc_b_layer_{}",i))
        return(S_in, enc_w, enc_b)

    # PC-AE Encoder
    def pcae_encoder(S_in, enc_w, enc_b, latent_layer):
        ''' Graph of the PC-AE encoder

        Input:
          - S_in: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
          - enc_w: Dictionary with tensors (filters) for the convolution layers.
          Type: <dictionary[enc_w_layer_{},...,enc_w_layer_{}]>
          - enc_b: Dictionary with the bias tensors for the convolution layers. 
          Type: <dictionary[enc_b_layer_{},...,enc_b_layer_{}]>
          - latent_layer: number of latent variables. Type: int

        Output:
          - Z: Tensor with the latent representation. 
          Type <tensor (-1, latent_layer, 1)>
        '''
    
        # Initialize empty dictionaries for assigning the layers
        enc_layer = {}
        #actv_conv_op = {}
        # Assign initial (n-1) convolution layers
        for i in range(len(enc_w)-1):
            # First layer, dependent on the dimensionality of the input
            #point clouds
            if i == 0:
                enc_layer[i] = tf.nn.relu(
                                       tf.nn.conv1d(S_in, enc_w[i],\
                                                    stride=(1), padding="SAME")\
                                       + enc_b[i],\
                                        name="enclayer_0")

            # Further intermediate layers
            else:
                enc_layer[i] = tf.nn.relu(
                                       tf.nn.conv1d(enc_layer[i-1],
                                                    enc_w[i], stride=(1),
                                                    padding="SAME")
                                       + enc_b[i], 
                                       name=str.format("enclayer_{}", i))

        # Last convolution layer
        i = len(enc_w)-1
        enc_layer[i] =  tf.math.tanh(tf.nn.conv1d(enc_layer[i-1], 
                                                  enc_w[i], stride=(1),
                                                  padding="SAME")
                                        + enc_b[i],
                                        name=str.format("enclayer_{}", i))
    
        # Max pooling operation
        max_pool = tf.reduce_max(enc_layer[i], axis=1)
        
        # Extracting the latent representations
        Z = tf.reshape(max_pool, shape=(-1, latent_layer, 1),\
                       name="Z")
        
        # Output
        return(Z)

    # PC-AE Decoder parameters
    def pcae_dec_params(latent_layer, decoder_sizes):
        ''' Parameters of the PC-AE decoder

        Input:
          - latent_layer: Number of latent variables. Type: int
          - decoder_sizes: Array with the number of hidden neurons 
          for each fully connected layer. Type: <array (-1)>

        Output:
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>
        '''
        dec_w = {}
        dec_b = {}
        ## Decoder: Fully Connected Layers
        # First fully connected layer
        dec_w[0] = tf.Variable(tf.random_normal(
                                                [decoder_sizes[0],
                                                 latent_layer], mean=0.0, stddev=0.01, dtype=tf.float32, seed=0),
                                          name=str.format("dec_w_layer_{}", 0))
        dec_b[0] = tf.Variable(tf.random_normal(
                                                [decoder_sizes[0], 3],
                                                mean=0.0, stddev=0.01,
                                                dtype=tf.float32, seed=0),
                                          name=str.format("dec_b_layer_{}",0))
        # Further layers
        for i in range(1, len(decoder_sizes)):
            dec_w[i] = tf.Variable(tf.random_normal(
                                                    [decoder_sizes[i],
                                                     decoder_sizes[i-1]],
                                                    mean=0.0, stddev=0.01,
                                                    dtype=tf.float32, seed=0),
                                          name=str.format("dec_w_layer_{}", i))
            dec_b[i] = tf.Variable(tf.random_normal(
                                                    [decoder_sizes[i],3],
                                                    mean=0.0, stddev=0.01,
                                                    dtype=tf.float32, seed=0),
                                          name=str.format("dec_b_layer_{}",i))
        # Output
        return(dec_w, dec_b)

    ## PC-AE Decoder
    def pcae_decoder(latent_vec, decoder_sizes, dec_w, dec_b):
        ''' Graph of the PC-AE decoder

        Input:
          - latent_vec: tensor with the latent representations corresponding to 
          the input S_in. type: <tensor (-1, latent_layer, 1)>
          - decoder_sizes: array with the number of hidden neurons for 
          each fully connected layer. Type: <array (-1)>
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>

        Output:
          - S_out: Reconstructed  point cloud. 
          Type <tensor (-1, pc_size, 3)>    
        '''
        
        dec_layers = {}
        # Adapt code for 3 Dimensions
        latent_rep_trip = tf.concat((latent_vec, latent_vec, latent_vec), 2)
    
        # Fully Connected Layers
        for i in range(len(decoder_sizes)-1):
            # First layer after the latent representation
            if i == 0:
                dec_layers[i] = tf.nn.relu(
                                       tf.scan(
                                            lambda a, z: tf.add(
                                                          tf.matmul(dec_w[i],
                                                                    z),
                                                            dec_b[i]),
                                            latent_rep_trip, 
                                            initializer=tf.zeros(
                                                            (decoder_sizes[i],
                                                             3))
                                            ),
                                       name=str.format("declayer_{}", 0))

            # Intermediate layers, before the output
            else:
                dec_layers[i] = tf.nn.relu(
                                      tf.scan(
                                          lambda a, z: tf.add(
                                                        tf.matmul(dec_w[i], z),
                                                          dec_b[i]), 
                                          dec_layers[i-1],
                                          initializer=tf.zeros(
                                                             (decoder_sizes[i],
                                                              3))
                                      ),
                                name=str.format("declayer_{}", i))
        
        # Last layer, with the same shape as the input tensor x
        i = len(decoder_sizes)-1
        dec_layers[i] = tf.nn.sigmoid(
                                 tf.scan(
                                     lambda a, z: tf.add(
                                                    tf.matmul(dec_w[i], z),
                                                    dec_b[i]), 
                                    dec_layers[i-1],
                                    initializer=tf.zeros(
                                                       (decoder_sizes[i], 3)))
                                    )
    
        # Output tensor: retrieved point clouds
        S_out = tf.reshape(dec_layers[i], 
                        shape=(-1, decoder_sizes[i], 3), name="S_out")
        return(S_out)

    ## PC-AE Architecture
    def pcae(encoder_sizes, pc_size, latent_layer, decoder_sizes):
        ''' Builds the graph of the complete PC-AE architecture

        Input:
          - encoder_sizes: Array with the number of features for each 
          convolution layer. Type <array (-1)>
          - pc_size: Number of points in the point clouds. Type: int
          - latent_layer: Number of latent variables. Type: int
          - decoder_sizes: array with the number of hidden neurons for 
          each fully connected layer. Type: <array (-1)>

        Output:
          - S_in: Input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
          - Z: Tensor with the latent representation. 
          Type <tensor (-1, latent_layer, 1)>
          - S_out: Reconstructed  point cloud. Type <tensor (-1, pc_size, 3)>
          - enc_w: Dictionary with tensors (filters) for the convolution layers.
          Type: <dictionary[enc_w_layer_{},...,enc_w_layer_{}]>
          - enc_b: Dictionary with the bias tensors for the convolution layers. 
          Type: <dictionary[enc_b_layer_{},...,enc_b_layer_{}]>
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>
        '''
        S_in, enc_w, enc_b = PC_AE.pcae_enc_params(encoder_sizes, pc_size)
        Z = PC_AE.pcae_encoder(S_in, enc_w, enc_b, latent_layer)
        dec_w, dec_b = PC_AE.pcae_dec_params(latent_layer, decoder_sizes)
        S_out = PC_AE.pcae_decoder(Z, decoder_sizes, dec_w, dec_b)
        return(S_in, Z, S_out, enc_w, enc_b, dec_w, dec_b)

# 3D Point Cloud Variational Autoencoder (architecture)
class PC_VAE:
    ## PC-VAE encoder parameters
    def pcvae_enc_params(encoder_sizes, pc_size):
        ''' Initial values of the PC-VAE encoder parameters

        Input:
          - encoder_sizes: Array with the number of features for each 
          convolutional layer. Type <array (-1)>
          - pc_size: Number of points in the point clouds. Type: int

        Output:
          - S_in: Input placeholder of the PC-VAE. Type: <tensor (-1,pc_size,3)>
          - keep_prob: Input placeholder of the keeping probability for the 
          dropout in the last convolutional layer. Type: <tensor (1)>
          - enc_w: Dictionary with the weights for the convolutional layers,
          type: <dictionary[enc_w_layer_0, ..., enc_w_layer_mu,
                           enc_w_layer_sigma]>
          - enc_b: Dictionary with the biases for the convolutional layers,
          type: <dictionary[enc_b_layer_0, ..., enc_b_layer_mu,
                           enc_b_layer_sigma]>
        '''
    
        ## Input point cloud, placeholder
        S_in = tf.placeholder(tf.float32, (None, pc_size, 3), name="S_in")
    
        ## Drop rate, placeholder
        keep_prob = tf.placeholder(tf.float32, name="do_rate")
        
        ## Assign weights of the convolution layers
        enc_w = {}
        enc_b = {}
        for i in range(len(encoder_sizes)):
            # First layer: dependent on the size of the point clouds and fixed
            # number of features (3)
            if i == 0:
                # Convolution weights
                enc_w[i] = tf.Variable(
                                    tf.random_normal([1, 3, encoder_sizes[i]],
                                                      mean=0.0, stddev=0.01,
                                                      dtype=tf.float32, seed=0),
                                    name=str.format("enc_w_layer_{}", i))
                # Bias
                enc_b[i] = tf.Variable(
                                    tf.random_normal([encoder_sizes[i]],
                                                      mean=0.0, stddev=0.01,
                                                      dtype=tf.float32, seed=0),
                                    name=str.format("enc_b_layer_{}", i))
            # Further layers
            else:
                # Convolution weights
                enc_w[i] = tf.Variable(
                                    tf.random_normal([1, encoder_sizes[i-1],
                                                      encoder_sizes[i]],
                                                     mean=0.0, stddev=0.01,
                                                     dtype=tf.float32, seed=0),
                                    name=str.format("enc_w_layer_{}", i))
                # Bias
                enc_b[i] = tf.Variable(
                                    tf.random_normal([encoder_sizes[i]],
                                                     mean=0.0, stddev=0.01,
                                                     dtype=tf.float32, seed=0),
                                    name=str.format("enc_b_layer_{}", i))
            # Mu and sigma layers
            i = len(encoder_sizes)-1
            # Mu: Convolution weights
            enc_w[i+1] = tf.Variable(
                                tf.random_normal([1, encoder_sizes[i],
                                                  encoder_sizes[i]],
                                                 mean=0.0, stddev=0.01,
                                                 dtype=tf.float32, seed=0),
                                name="enc_w_layer_mu")
            # Mu: Bias
            enc_b[i+1] = tf.Variable(
                                tf.random_normal([encoder_sizes[i]],
                                                 mean=0.0, stddev=0.01,
                                                 dtype=tf.float32, seed=0),
                                name="enc_b_layer_mu")
            # sigma: Convolution weights
            enc_w[i+2] = tf.Variable(
                                tf.random_normal([1, encoder_sizes[i],
                                                  encoder_sizes[i]],
                                                 mean=0.0, stddev=0.01,
                                                 dtype=tf.float32, seed=0),
                                name="enc_w_layer_sigma")
            # sigma: Bias
            enc_b[i+2] = tf.Variable(
                                tf.random_normal([encoder_sizes[i]],
                                                 mean=0.0, stddev=0.01,
                                                 dtype=tf.float32, seed=0),
                                name="enc_w_layer_sigma")

        # Output
        return (S_in, keep_prob, enc_w, enc_b)

    # PC-VAE Encoder
    def pcvae_encoder(S_in, enc_w, enc_b, latent_layer, keep_prob):
        ''' Graph of the PC-VAE encoder

        Input:
          - S_in: Input placeholder of the PC-VAE. Type: <tensor (-1,pc_size,3)>
          - enc_w: Dictionary with the weights for the convolutional layers.
          Type: dictionary[enc_w_layer_0, ..., enc_w_layer_mu,
                           enc_w_layer_sigma]
          - enc_b: Dictionary with the biases for the convolutional layers.
          Type: dictionary[enc_b_layer_0, ..., enc_b_layer_mu,
                           enc_b_layer_sigma]
          - latent_layer: Number of latent variables. Type: int
          - keep_prob: Tensor with (1-drop_out ratio). Type: <tensor (1)>

        Output:
          - Z: Latent representation. Type:
                          <tensor (batch_size, latent_layer, 1)>
          - mu_encoder: Tensor with the mean values to sample the latent
          representations. Type: <tensor (batch_size, latent_layer, 1)
          - logvar_encoder: Tensor with the standard deviation
            values to sample the latent representations.
            Type: <tensor (batch_size, latent_layer, 1)
        '''
    
        # Initialize empty dictionaries for assigning the layers
        enc_layer = {}
        
        # Assign initial (n-1) convolution layers

        for i in range(len(list(enc_w.keys()))-3):
            # First layer, dependent on the dimensionality of the input
            # point clouds
            print(i)
            if i == 0:
                # 1D convolution, batch normalization, leaky_relu
                enc_layer[i] = tf.nn.leaky_relu(
                                  tf.layers.batch_normalization(
                                     tf.nn.conv1d(S_in, enc_w[i], 
                                                   stride=(1), padding="SAME")
                                      + enc_b[i], momentum=0.9, training=True),
                                  name="enclayer_0")
    
            # Further intermediate layers
            else:
                enc_layer[i] = tf.nn.leaky_relu(
                                   tf.layers.batch_normalization(
                                      tf.nn.conv1d(enc_layer[i-1], enc_w[i],
                                                   stride=(1), padding="SAME")
                                      + enc_b[i], momentum=0.9, training=True),
                                  name=str.format("enclayer_{}", i))

        # Last convolutional layer
        i = len(enc_w)-3
        enc_layer[i] = tf.nn.tanh(
                           tf.layers.batch_normalization(
                              tf.nn.conv1d(enc_layer[i-1], enc_w[i],
                                           stride=(1), padding="SAME")
                              + enc_b[i], momentum=0.9),
                          name=str.format("enclayer_{}", i))
    
        # Max pooling layer
        max_pool = tf.reshape(tf.reduce_max(enc_layer[i], axis=1),
                              shape=(-1, 1, latent_layer))
    
        # Mu (mean) layer
        mu_encoder =  tf.identity(
                          tf.nn.dropout(
                            tf.nn.conv1d(max_pool, enc_w[i+1], stride=(1),
                                         padding="SAME")
                            + enc_b[i+1], keep_prob=keep_prob),
                            name=str.format("mu", i))
    
        # Sigma (std. dev) layer
        logvar_encoder = tf.nn.sigmoid(
                         tf.nn.conv1d(max_pool, enc_w[i+2], stride=(1),
                                      padding="SAME")
                         + enc_b[i+2], name=str.format("sigma", i))
    
        # re-parameterization trick
        # Normal distribution
        N = tf.random.normal(shape=tf.shape(logvar_encoder),
                               mean=0, stddev=1, dtype=tf.float32, seed=0,
                               name=str.format("random_eps_{}", i))
        # Deviation
        std_encoder = tf.exp(logvar_encoder * 0.5)
    
        # Extracting the latent representations
        # print(mu_encoder.shape, std_encoder.shape, N.shape)
        # exit()
        Z = tf.reshape(mu_encoder + tf.multiply(std_encoder, N), 
                       shape=(-1, latent_layer, 1),
                       name="Z")
        # Output
        return (Z, mu_encoder, logvar_encoder)

    # PC-VAE Decoder parameters
    def pcvae_dec_params(latent_layer, decoder_sizes):
        ''' Parameters of the PC-VAE decoder

        Input:
          - latent_layer: Number of latent variables. Type: int
          - decoder_sizes: Array with the number of hidden neurons 
          for each fully connected layer. Type: <array (-1)>

        Output:
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>
        '''
        dec_w = {}
        dec_b = {}
        ## Decoder: Fully Connected Layers
        # First fully connected layer
        dec_w[0] = tf.Variable(tf.random_normal(
                                                [decoder_sizes[0],
                                                 latent_layer], mean=0.0, stddev=0.01, dtype=tf.float32, seed=0),
                                          name=str.format("dec_w_layer_{}", 0))
        dec_b[0] = tf.Variable(tf.random_normal(
                                                [decoder_sizes[0], 3],
                                                mean=0.0, stddev=0.01,
                                                dtype=tf.float32, seed=0),
                                          name=str.format("dec_b_layer_{}",0))
        # Further layers
        for i in range(1, len(decoder_sizes)):
            dec_w[i] = tf.Variable(tf.random_normal(
                                                    [decoder_sizes[i],
                                                     decoder_sizes[i-1]],
                                                    mean=0.0, stddev=0.01,
                                                    dtype=tf.float32, seed=0),
                                          name=str.format("dec_w_layer_{}", i))
            dec_b[i] = tf.Variable(tf.random_normal(
                                                    [decoder_sizes[i],3],
                                                    mean=0.0, stddev=0.01,
                                                    dtype=tf.float32, seed=0),
                                          name=str.format("dec_b_layer_{}",i))
        # Output
        return(dec_w, dec_b)

    # PC-VAE Decoder
    def pcvae_decoder(Z, decoder_sizes, dec_w, dec_b):
        ''' Graph of the PC-VAE decoder

        Input:
          - latent_vec: tensor with the latent representations corresponding to 
          the input S_in. type: <tensor (-1, latent_layer, 1)>
          - decoder_sizes: array with the number of hidden neurons for 
          each fully connected layer. Type: <array (-1)>
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>

        Output:
          - S_out: Reconstructed  point cloud. 
          Type <tensor (-1, pc_size, 3)>   
        '''
        dec_layers = {}
        # Adapt code for 3 Dimensions
        latent_rep_trip = tf.concat((Z, Z, Z), 2)
    
        # Fully Connected Layers
        for i in range(len(decoder_sizes) - 1):
            # First layer after the latent representation
            if i == 0:
                dec_layers[i] = tf.nn.leaky_relu(
                                  tf.scan(
                                     lambda a, z: 
                                       tf.layers.batch_normalization(
                                          tf.add(
                                             tf.matmul(
                                                dec_w[i],z),
                                             dec_b[i]), 
                                          momentum=0.9, training=True), 
                                       latent_rep_trip, 
                                       initializer=tf.zeros(
                                                      (decoder_sizes[i], 3))),
                                  name=str.format("declayer_{}", 0))
    
            # Intermediate layers, before the output
            else:
                dec_layers[i] = tf.nn.leaky_relu(
                                   tf.scan(
                                      lambda a, z:
                                        tf.layers.batch_normalization(
                                           tf.add(
                                              tf.matmul(
                                               dec_w[i], z),
                                              dec_b[i]),
                                           momentum=0.9, training=True), 
                                        dec_layers[i-1], 
                                        initializer=tf.zeros(
                                                      (decoder_sizes[i], 3))),
                                  name=str.format("declayer_{}", i))
    
        # Last layer, with the same shape as the input tensor x
        i = len(decoder_sizes) - 1
        dec_layers[i] = tf.nn.sigmoid(
                           tf.scan(
                              lambda a, z: 
                                tf.add(
                                   tf.matmul(
                                      dec_w[i], z),
                                   dec_b[i]),
                                dec_layers[i-1],
                              initializer=tf.zeros((decoder_sizes[i], 3))))
    
        # Output tensor: retrieved point clouds
        S_out = tf.reshape(dec_layers[i], 
                        shape=(-1, decoder_sizes[i], 3),
                        name="S_out")
        # Output
        return(S_out)

    # PC-VAE Architecture
    def pcvae(encoder_sizes, pc_size, latent_layer, decoder_sizes):
        ''' Builds the graph of the complete PC-VAE architecture

        Input:
          - encoder_sizes: Array with the number of features for each 
          convolution layer. Type <array (-1)>
          - pc_size: Number of points in the point clouds. Type: int
          - latent_layer: Number of latent variables. Type: int
          - decoder_sizes: array with the number of hidden neurons for 
          each fully connected layer. Type: <array (-1)>

        Output:
          - S_in: Input placeholder of the PC-AE, type: tensor, (-1,pc_size,3) 
          - Z: Tensor with the latent representation. 
          Type <tensor (-1, latent_layer, 1)> 
          - S_out: Reconstructed  point cloud. Type <tensor (-1, pc_size, 3)> 
          - mu: Tensor with the mean values for sampling the latent
           representation. Type <tensor (-1, latent_layer, 1)>
          - logvar: Tensor with the standard deviation for sampling the latent
           representation. Type <tensor (-1, latent_layer, 1)>
          - k_dout: Placeholder for assigning the dropout ratio. 
          Type <Tensor (1)>
          - enc_w: Dictionary with tensors (filters) for the convolution layers.
          Type: <dictionary[enc_w_layer_{},...,enc_w_layer_{}]>
          - enc_b: Dictionary with the bias tensors for the convolution layers. 
          Type: <dictionary[enc_b_layer_{},...,enc_b_layer_{}]>
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>
        '''
        S_in, k_dout, enc_w, enc_b = PC_VAE.pcvae_enc_params(encoder_sizes,
                                                            pc_size)
        Z, mu, logvar = PC_VAE.pcvae_encoder(S_in, enc_w, enc_b, latent_layer, 
                                             k_dout)
        dec_w, dec_b = PC_VAE.pcvae_dec_params(latent_layer, decoder_sizes)
        S_out = PC_VAE.pcvae_decoder(Z, decoder_sizes, dec_w, dec_b)

        return(S_in, Z, S_out, mu, logvar, k_dout, enc_w, enc_b, dec_w, dec_b)

# Point2FFD
class Point2FFD:
    # Point2FFD encoder parameters
    def p2ffd_encoder_params(encoder_sizes, pc_size):
        ''' Initial values of Point2FFD parameters

        Input:
          - encoder_sizes: Array with the number of features for each 
          convolution layer. Type <array (-1)>
          - pc_size: Number of points in the point clouds. Type: int
        
        Output:
          - S_in: Input placeholder. Type <tensor (-1,pc_size,3)>
          - enc_w: Dictionary with tensors (filters) for the convolution layers.
          Type: <dictionary[enc_w_layer_{},...,enc_w_layer_{}]>
          - enc_b: Dictionary with the bias tensors for the convolution layers. 
          Type: <dictionary[enc_b_layer_{},...,enc_b_layer_{}]>
          - gamma_noise: Placeholder for the parameter "gamma" that enables
          the noise in the latent layer. Type <tensor ()>
        '''
    
        ## Input point cloud, placeholder
        S_in = tf.placeholder(tf.float32, (None, pc_size, 3), name="S_in")

        ## Noise for the latent variables
        gamma_noise = tf.placeholder(tf.float32, (), name="gamma_n")

        ## Assign convolutional layers
        enc_w = {}
        enc_b = {}
        for i in range(len(encoder_sizes)):
            # First layer: dependent on the size of the point clouds and fixed
            #number of features (3)
            if i == 0:
                # Convolutional weights
                enc_w[i] = tf.Variable(\
                    tf.random_normal(\
                    [1, 3, encoder_sizes[i]], mean=0.0, stddev=0.1, \
                        dtype=tf.float32, seed=0), name=str.format(\
                        "enc_w_layer_{}", i))
                # Bias
                enc_b[i] = tf.Variable(\
                    tf.random_normal(\
                        [encoder_sizes[i]],\
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                    name=str.format("enc_b_layer_{}",i))
            # Further layers
            else:
                # Convolutional weights
                enc_w[i] = tf.Variable(\
                    tf.random_normal(\
                    [1, encoder_sizes[i-1], encoder_sizes[i]], \
                        mean=0.0, stddev=0.1, \
                        dtype=tf.float32, seed=0), name=str.format(\
                        "enc_w_layer_{}", i))
                # Bias
                enc_b[i] = tf.Variable(\
                    tf.random_normal(\
                        [encoder_sizes[i]],\
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                    name=str.format("enc_b_layer_{}",i))
        
        return(S_in, enc_w, enc_b, gamma_noise)

    # Point2FFD Encoder graph
    def p2ffd_encoder(S_in, enc_w, enc_b, latent_layer,
                          gamma_n=tf.constant(0), sigma_n=0.3882):
        ''' Graph of the Point2FFD encoder

        Input:
          - S_in: input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
          - enc_w: Dictionary with tensors (filters) for the convolution layers.
          Type: <dictionary[enc_w_layer_{},...,enc_w_layer_{}]>
          - enc_b: Dictionary with the bias tensors for the convolution layers. 
          Type: <dictionary[enc_b_layer_{},...,enc_b_layer_{}]>
          - latent_layer: number of latent variables. Type: int
          - gamma_n: Placeholder for the parameter "gamma" that enables
          the noise in the latent layer. Type <tensor ()>
          - sigma_n: standard deviation for generating the Gaussian noise.
          Type: <float, ()>

        Output:
          - Z: Tensor with the latent representation. 
          Type <tensor (-1, latent_layer, 1)>
        '''
    
        # Initialize empty dictionaries for assigning the layers
        enc_layer = {}
        #actv_conv_op = {}
        # Assign initial (n-1) convolution layers
        for i in range(len(enc_w)-1):
            # First layer, dependent on the dimensionality of the input
            #point clouds
            if i == 0:
                enc_layer[i] = tf.nn.relu(
                                       tf.nn.conv1d(S_in, enc_w[i],\
                                                    stride=(1), padding="SAME")\
                                       + enc_b[i],\
                                        name="enclayer_0")

            # Further intermediate layers
            else:
                enc_layer[i] = tf.nn.relu(
                                       tf.nn.conv1d(enc_layer[i-1],
                                                    enc_w[i], stride=(1),
                                                    padding="SAME")
                                       + enc_b[i], 
                                       name=str.format("enclayer_{}", i))

        # Last convolution layer
        i = len(enc_w)-1
        enc_layer[i] =  tf.math.tanh(tf.nn.conv1d(enc_layer[i-1], 
                                                  enc_w[i], stride=(1),
                                                  padding="SAME")
                                        + enc_b[i],
                                        name=str.format("enclayer_{}", i))
    
        # Max pooling operation
        max_pool = tf.reduce_max(enc_layer[i], axis=1)

        # Add noise with normal distribution
        Z = tf.reshape(
                         tf.scan(
                             lambda a,z: z + gamma_n*tf.random_normal(
                                     shape=[latent_layer,], mean=z,
                                     stddev=sigma_n,
                                     dtype=tf.float32, seed=0),
                                 max_pool, initializer=tf.zeros([latent_layer,])
                            ), shape=(-1, latent_layer, 1), 
                        name="Z")
        
        # Output
        return(Z)

    # Shape Classifier parameters
    def p2ffd_classifier_params(latent_layer, class_sizes):
        ''' Initial values of the classifier parameters

        Input:
          - latent_layer: number of latent variables. Type: int
          - class_sizes: Array with the number of features for each 
          classifier layer. Type <array (-1)>

        Output:
          - class_w: Dictionary with the layers' weights.
          Type: <dictionary[class_w_layer_{},...,class_w_layer_{}]>
          - class_b: Dictionary with the layers' bias.
          Type: <dictionary[class_b_layer_{},...,class_b_layer_{}]>
        '''

        class_w = {}
        class_b = {}
        ## MLP Layers
        # First layer
        class_w[0] = tf.Variable(tf.random_normal(\
            [latent_layer, class_sizes[0]], \
                mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                name=str.format("class_w_layer_{}", 0))
        class_b[0] =  tf.Variable(tf.random_normal(\
            [class_sizes[0]],\
            mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
            name=str.format("class_b_layer_{}", 0))
    
        # Additional layers
        for i in range(1, len(class_sizes)):
            class_w[i] = tf.Variable(tf.random_normal(\
                [class_sizes[i-1], class_sizes[i]], \
                    mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                    name=str.format("class_w_layer_{}",i))
            class_b[i] = tf.Variable(tf.random_normal(\
                [class_sizes[i]],\
                 mean=0.0, stddev=0.1, dtype=tf.float32, seed=0),\
                 name=str.format("class_b_layer_{}",i))
    
        return(class_w, class_b)

    # Shape classifier graph
    def p2ffd_classifier(Z, class_w, class_b):
        ''' Graph of the Point2FFD Classifier
        Inputs:
          - Z: Tensor with the latent representation. 
          Type <tensor (-1, latent_layer, 1)>
          - class_w: Dictionary with the layers' weights.
          Type: <dictionary[class_w_layer_{},...,class_w_layer_{}]>
          - class_b: Dictionary with the layers' bias.
          Type: <dictionary[class_b_layer_{},...,class_b_layer_{}]>

        Outputs:
          - class_labl: Tensor with the class label. 
          Type <tensor ()>
          - class_prob: Tensor with the selection probabilities. 
          Type <tensor (-1, latent_layer)>
        '''
        # Adapt code for nfeat Dimensions
       
        latent_rep = tf.reshape(Z, shape=(-1, Z.shape[1]))
        # Calculate the number of layer
        n_layers = int(len(class_w.keys()))
    
        layer_res = {}
        for i in range(n_layers):
            if i == 0:
                layer_res[i] = \
                    tf.nn.relu(\
                        tf.add(\
                            tf.matmul(latent_rep, class_w[i]),
                            class_b[i]),
                        name=str.format("mlpclass_{}", i))
            else:
                layer_res[i] = \
                    tf.nn.relu(\
                        tf.add(\
                            tf.matmul(layer_res[i-1],\
                            class_w[i]),
                        class_b[i]),
                        name=str.format("mlpclass_{}", i))
        # Last layer
        layer_res[n_layers-1] = tf.nn.sigmoid(
                                   tf.add(
                                      tf.matmul(layer_res[n_layers-2],\
                                         class_w[n_layers-1]),
                                      class_b[n_layers-1],
                                name=str.format("mlpclass_{}", n_layers-1)))
    
        #yclass = tf.nn.softmax(layer_res[str(n_layers-1)])
        class_labl = tf.math.argmax(layer_res[n_layers-1], axis=1, 
                                    output_type=tf.dtypes.int64,
                                    name="class_labl")
        class_prob = tf.nn.softmax(layer_res[n_layers-1],
                                   name="class_prob")
    
        # Output
        return(class_labl, class_prob)
    
    # Decoder
    def p2ffd_decoder_params(latent_layer, decoder_sizes):
        ''' Parameters of the Poin2FFD decoder

        Input:
          - latent_layer: Number of latent variables. Type: int
          - decoder_sizes: Array with the number of hidden neurons 
          for each fully connected layer. Type: <array (-1)>

        Output:
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>
        '''
        dec_w = {}
        dec_b = {}
        ## Decoder: Fully Connected Layers
        # First fully connected layer
        dec_w[0] = tf.Variable(tf.random_normal(
                                                [decoder_sizes[0],
                                                 latent_layer], mean=0.0, stddev=0.01, dtype=tf.float32, seed=0),
                                          name=str.format("dec_w_layer_{}", 0))
        dec_b[0] = tf.Variable(tf.random_normal(
                                                [decoder_sizes[0], 3],
                                                mean=0.0, stddev=0.01,
                                                dtype=tf.float32, seed=0),
                                          name=str.format("dec_b_layer_{}",0))
        # Further layers
        for i in range(1, len(decoder_sizes)):
            dec_w[i] = tf.Variable(tf.random_normal(
                                                    [decoder_sizes[i],
                                                     decoder_sizes[i-1]],
                                                    mean=0.0, stddev=0.01,
                                                    dtype=tf.float32, seed=0),
                                          name=str.format("dec_w_layer_{}", i))
            dec_b[i] = tf.Variable(tf.random_normal(
                                                    [decoder_sizes[i],3],
                                                    mean=0.0, stddev=0.01,
                                                    dtype=tf.float32, seed=0),
                                          name=str.format("dec_b_layer_{}",i))
        # Output
        return(dec_w, dec_b)

    # PC-AE Decoder
    def p2ffd_decoder(Z, decoder_sizes, dec_w, dec_b):
        ''' Graph of the Point2FFD decoder

        Input:
          - Z: tensor with the latent representations corresponding to 
          the input S_in. type: <tensor (-1, latent_layer, 1)>
          - decoder_sizes: array with the number of hidden neurons for 
          each fully connected layer. Type: <array (-1)>
          - dec_w: Dictionary with the weights of the decoder layers. 
          Type: <dictionary[dec_w_layer_0,...,dec_w_layer_{}]>
          - dec_b: Dictionary with the biases of the decoder layers. 
          Type: <dictionary[dec_b_layer_0,...,dec_b_layer_{}]>

        Output:
          - V_def: Lattice deformations. 
          Type <tensor (-1, pc_size, 3)>    
        '''
        
        dec_layers = {}
        # Adapt code for 3 Dimensions
        Z_concat = tf.concat((Z, Z, Z), 2)
    
        # Fully Connected Layers
        for i in range(len(decoder_sizes)-1):
            # First layer after the latent representation
            if i == 0:
                dec_layers[i] = tf.nn.relu(
                                       tf.scan(
                                            lambda a, z: tf.add(
                                                          tf.matmul(dec_w[i],
                                                                    z),
                                                            dec_b[i]),
                                            Z_concat, 
                                            initializer=tf.zeros(
                                                            (decoder_sizes[i],
                                                             3))
                                            ),
                                       name=str.format("declayer_{}", 0))

            # Intermediate layers, before the output
            else:
                dec_layers[i] = tf.nn.relu(
                                      tf.scan(
                                          lambda a, z: tf.add(
                                                        tf.matmul(dec_w[i], z),
                                                          dec_b[i]), 
                                          dec_layers[i-1],
                                          initializer=tf.zeros(
                                                             (decoder_sizes[i],
                                                              3))
                                      ),
                                name=str.format("declayer_{}", i))
        
        # Last layer, with the same shape as the input tensor x
        i = len(decoder_sizes)-1
        dec_layers[i] = tf.nn.sigmoid(
                                 tf.scan(
                                     lambda a, z: tf.add(
                                                    tf.matmul(dec_w[i], z),
                                                    dec_b[i]), 
                                    dec_layers[i-1],
                                    initializer=tf.zeros(
                                                       (decoder_sizes[i], 3)))
                                    )
    
        # Output tensor: Displacement of the control points
        V_def = tf.reshape(dec_layers[i], 
                        shape=(-1, decoder_sizes[i], 3), name="V_def")
        return(V_def)

    # FFD operator
    def p2ffd_ffdop(B_ffd, V0_ffd, V_def, class_labl):
        ''' Graph of the FFD operator

        Input:
          - B_ffd: 3D array with the Bernstein polynomial coefficients for
          each mesh template. 
          Type: <array (n_templates, pc_size, n_ffd_control_points)>
          - V0_ffd: 3D array with the control points' coordinates for
          each template lattice. 
          Type: <array (n_templates, n_ffd_control_points, 3)>
          - V_def: Tensor with the predicted deformation of the control points.
          Type: <tensor (-1, n_ffd_control_points, 3)>
          - class_lab: Tensor with the labels of the selected prototypes.
          Type: <tensor (-1, 1)>

        Output:
          - S_out: Reconstructed  point clouds. 
          Type <tensor (-1, pc_size, 3)>   
        '''

        # Arange the Bernstein polynomial matrices
        B = tf.gather(tf.constant(B_ffd, dtype=tf.float32), 
                      class_labl, name="B")
        # Arange the deformed control points
        CP = tf.add(V_def, tf.gather(
                              tf.constant(V0_ffd, dtype=tf.float32), 
                              class_labl), 
                    name="Vd")
        # Free-form deformation
        pc_ffd = tf.scan(lambda a,z: tf.matmul(z[0], z[1]), [B,CP],\
            initializer=tf.zeros((B.shape[1], 3)))
        S_out = tf.reshape(pc_ffd, (-1, B.shape[1], 3), name="S_out")

        return(S_out)

    # Point2FFD Architecture
    def point2ffd(B, V0, encoder_sizes, pc_size, latent_layer, class_sizes, 
                  decoder_sizes, sigma_n=0.3882):
        ''' Builds the graph of the complete PC-AE architecture

        Input:
          - B: 3D array with the Bernstein polynomial coefficients for
          each mesh template. 
          Type: <array (n_templates, pc_size, n_ffd_control_points)>
          - V0: 3D array with the control points' coordinates for
          each template lattice. 
          Type: <array (n_templates, n_ffd_control_points, 3)>
          - encoder_sizes: Array with the number of features for each 
          convolution layer. Type <array (-1)>
          - pc_size: Number of points in the point clouds. Type: int
          - latent_layer: Number of latent variables. Type: int
          - decoder_sizes: array with the number of hidden neurons for 
          each fully connected layer. Type: <array (-1)>
          - sigma_n:

        Output:
          - S_in: Input placeholder of the PC-AE, type: tensor, (-1,pc_size,3)
          - Z: Tensor with the latent representation. 
          Type <tensor (-1, latent_layer, 1)>
          - S_out: Reconstructed  point cloud. Type <tensor (-1, pc_size, 3)>
          - class_labl:
          - class_prob: 
        '''
        # Encoder
        S_in, enc_w, enc_b, gamma_noise = Point2FFD.\
            p2ffd_encoder_params(encoder_sizes, pc_size)
        Z = Point2FFD.p2ffd_encoder(S_in, enc_w, enc_b, latent_layer,
                          gamma_noise, sigma_n)
        # Classifier
        class_w, class_b = Point2FFD.\
                      p2ffd_classifier_params(latent_layer, class_sizes)
        class_labl, class_prob = Point2FFD.p2ffd_classifier(Z, class_w, class_b)
        # Decoder
        dec_w, dec_b = Point2FFD.p2ffd_decoder_params(latent_layer,
                                                      decoder_sizes)
        V_def = Point2FFD.p2ffd_decoder(Z, decoder_sizes, dec_w, dec_b)
        S_out = Point2FFD.p2ffd_ffdop(B, V0, V_def, class_labl)

        return(S_in, Z, S_out, class_labl, class_prob, gamma_noise)

# Reconstruction losses
class losses:
    # Mean Squared Distance
    def msd_gpu(S_in, S_out):
        ''' Mean squared distance (MSD) measured for a batch of organized 
        3D point clouds. Function utilized for GPU computation.

        Input:
          - S_in: Input placeholder of the PC-VAE. Type: <tensor (-1,pc_size,3)>
          - S_out: Reconstructed  point cloud. Type <tensor (-1, pc_size, 3)>   
          
        Output:
          - loss: Tensor with the mean MSD for the input batch of point clouds.
          Type: <tensor (1)>
        '''
        loss = tf.reduce_mean(
                  tf.reduce_mean(
                     tf.norm((S_in - S_out), axis=2), 
                     axis=1))
        return(loss)

    # Kullback-Leibler Divergence
    def KLD(z_mu, z_sigma):
        ''' Kullback-Leibler Divergence (KLD) as utilized for training the 
        PC-VAE. Function utilized for GPU computation.

        Input:
          - z_mu: Tensor with the mean values for sampling the latent
           representation. Type <tensor (-1, latent_layer, 1)>
          - z_sigma: Tensor with the standard deviation for sampling the latent
           representation. Type <tensor (-1, latent_layer, 1)>

        Output:
          - KL_div_loss: Mean KLD measured for the input batch.
          Type: <tensor, (1)>
        '''
        KL_div_loss = -0.5*tf.reduce_sum(
                                1+z_sigma-tf.pow(z_mu,2)-
                                tf.exp(z_sigma), reduction_indices=1)
        KL_div_loss = tf.reduce_mean(KL_div_loss)
        return(KL_div_loss)

    # Chamfer Distance
    def CD(S_in, S_out):
        ''' Chamfer distance (CD) calculated between a pair of 3D point
        clouds. This implementation is not suitable for GPU computation.

        Input:
          - S_in: Reference 3D point cloud. Type: <array (N,3)>
          - S_out: Output 3D point cloud. Type: <array (N,3)>

        Output:
          - cd: Chamfer Distance measured between the two point clouds.
          Type: <float (1)>
        '''
        distm = cdist(S_in, S_out, "euclidean")**2
        cd = (np.sum(np.min(distm, axis=0))
                + np.sum(np.min(distm, axis=1)))
        return(cd)

    # Combined MSD (Point2FFD)
    def msd_comb(S_in, S_out, pcs_t, ind_t, class_prob):
        ''' Mean squared distance (MSD) measured for a batch of organized 
        3D point clouds and weighted by the selection probability of the
        templates (Point2FFD).

        Input:
          - S_in: Input placeholder of the PC-VAE. Type: <tensor (-1,pc_size,3)>
          - S_out: Reconstructed  point cloud. Type <tensor (-1, pc_size, 3)>
          - pcs_t: 3D Point clouds that represent the mesh templates.
          Type <tensor (-1,pc_size,3)
          - ind_t: Indices of the selected template per input shape.
          Type <tensor (-1)>
          - class_prob: Selection probability of each template for each input 
          shape. Type <tensor (-1,number_of_templates)>
          
        Output:
          - loss: Tensor with the mean MSD for the input batch of point clouds.
          Type: <tensor (1)>
        '''

        msd_class = tf.reduce_sum(
                       tf.scan(
                          lambda a,z: tf.reduce_mean(
                                         (class_prob[:,z]/tf.reduce_sum\
                                                         (class_prob, axis=1)
                                         )*tf.reduce_mean(tf.norm((S_in - pcs_t[z,:,:]), axis=2), axis=1)), ind_t,\
                          initializer=tf.zeros(()))
                    )
        msd_rec = tf.reduce_mean(
                     tf.reduce_mean(
                        tf.norm((S_in - S_out), axis=2), axis=1))
        
        loss_comb = msd_class + msd_rec

        return(loss_comb)

# Class of algorithms for training the deep-generative models
class arch_training:
    # Data set loader
    def data_loader(config):
        ''' Subscript to load CAE data as 3D point clouds and generate
        the partitions for training and testing the deep-generative models.

        Input:
          - config: Dictionary with the settings for training a model. 
        Type: dictionary (check the documentation for more details).

        Output:
          - data_training: Partition with the point clouds for training the
          networks. Type: <array, (-1, N, 3)>
          - list_batches_training: List with arrays that contain the indices of
          the shapes utilized in each training batch. Type: <list (batch size)>
          - data_test: Partition with the point clouds for testing the
          networks. Type: <array, (-1, N, 3)>
          - list_batches_test: List with arrays that contain the indices of
          the shapes utilized in each test batch. Type: <list (batch size)>
          - pc_size: Size (N) of the point clouds. Type: int
          - samples_training: Array with the name of the files utilized for
          training the networks. Type: <array (-1)>
          - samples_test: Array with the name of the files utilized for
          testing the networks. Type: <array (-1)>
        '''
        ## Initialize
        # List of data set directories
        data_set_list = list(config["dataset"])
        # List with the directories that contain the files with the point
        # sampling probability (if available)
        prb_samp_list = list(config["probsamp"])
        # List with the number/ids of the shapes to be sampled per data set
        pointcld_list = list(config["shapelist"])
        # Size of the point clouds
        pc_size = int(config["pc_size"])

        ## Sampling
        data_set_dict = {}
        log_name = []
        # Loop over the data sets
        for i in range(len(data_set_list)):
            data_set_path = data_set_list[i]
            prob = prb_samp_list[i]
            pclds = pointcld_list[i]
            
            # Load data set
            data_temp, log_temp = CAE2PC.pc_sampling(data_set_path, pc_size,
                                                     n_pclouds=pclds, 
                                                     p_samp=prob)
            if log_temp[0] == "FLAG ERROR":
                print("ERROR: Loading data set")
                print("Check configuration file")
                return("FLAG_ERROR", True)
            data_set_dict[i] = data_temp
            log_name += log_temp
        
        # Concatenate data sampled from the data sets
        data_set = np.zeros((len(log_name),data_set_dict[0].shape[1],
                             data_set_dict[0].shape[2]))
        size_cnt = 0
        for i in range(len(data_set_dict.keys())):
            data_set[size_cnt:size_cnt+data_set_dict[i].shape[0],:,:] =\
                                                        data_set_dict[i]
            size_cnt += data_set_dict[i].shape[0]
        
        ## Shuffle and split the complete data set
        # Shuffle
        indices = list(range(len(log_name)))
        shuffle(indices)
        data_set = data_set[indices,:,:]
        log_name = np.array(log_name)[indices]
        # Split
        split_fraction = int(config["frac_training"]*len(log_name))
        data_training = data_set[:split_fraction,:,:]
        samples_training = log_name[:split_fraction]
        data_test = data_set[split_fraction:,:,:]
        samples_test = log_name[split_fraction:]
        
        ## Define batches
        # Training
        batch_training = int(config["training_batch_size"])
        list_batches_training = []
        indices_training = list(range(len(samples_training)))
        for i in range(0,len(samples_training),batch_training):
            list_batches_training.append(indices_training[i:i+batch_training])
        # Test
        batch_test= int(config["test_batch_size"])
        list_batches_test = []
        indices_test = list(range(len(samples_test)))
        for i in range(0,len(samples_test),batch_test):
            list_batches_test.append(indices_test[i:i+batch_test])
        
        ## Return samples
        return(data_training, list_batches_training, 
               data_test, list_batches_test, pc_size,
               samples_training, samples_test)            

    # Mesh template loader
    def meshtemp_loader(config, pc_size, normlimits):
        ''' Generates the FFD parameterizations of the templates
        
        Input:
        - config: Dictionary with the settings for training a model. 
        Type: dictionary (check the documentation for more details).
        - pc_size: Size (N) of the point clouds. Type: int
        - normlimits: Interval in which the input data is defined. If not
        specified, the range of the data is considered instead. Type: 
        <array, (2,)>

        Output:
        - B: Tensor with the coefficients of the tri-variate Bernstein
        polynomials for each template. 
        Type <array, (-1, pc_size, number_control_points)
        - V0: Tensor with the coordinates of the control points for each
        parameterized template. 
        Type <array, (n_templates,number_control_points,3)>
        - pcs_t: Tensor with the point clouds of the parameterized
        templates. Type <tensor, (n_templates,pc_size,3)>
        - ind_t: Tensor with the indeces of the templates.
        Type <tensor, (n_templates)>
        '''

        ## Load mesh templates
        # List with the location of the templates
        temp_list = list(config["temp_list"])
        # Number of templates
        ntemps = len(temp_list)
        # FFD lattice configuration (number of planes)
        L,M,N = list(config["ffd_lmn"])
        # Point cloud size
        pc_size = int(config["pc_size"])

        ## Generate FFD Matrices
        # Allocate a tensor for the tri-variate Bernstein polynomial
        #coefficients
        B = np.zeros((ntemps, pc_size, int(L*M*N)))
        # Allocate a tensor for the coordinates of the control points
        V0 = np.zeros((ntemps, L*M*N, 3))
        # Temporary allocation of point clouds and normalized coordinates
        pc_temp = np.zeros((ntemps, pc_size, 3))
        pc_n_tp = np.zeros((ntemps, pc_size, 3))

        for p in range(ntemps):
            ## Initial shape
            nodes = np.array(pd.read_csv("{}".format(temp_list[p]),
                                         header=None, sep=" "))[:,0:3]
            # Normalize to the data set span
            nodes = CAE2PC.data_set_norm(nodes, inp_lim=normlimits, 
                                  out_lim=np.array([0.1, 0.9]))[0]
            pc_temp[p, :, :] = nodes
            
            # Embbed in a FFD lattice
            nodes_n = np.array(nodes)
            for j in range(3): 
                nodes_n[:,j] = (nodes[:,j] - np.min(nodes[:,j]))/\
                    (np.max(nodes[:,j]) - np.min(nodes[:,j]))
            pc_n_tp[p,:,:] = nodes_n

            # Initial control points
            xc0 = np.linspace(
                     np.min(nodes[:,0])-0.01*np.abs(
                                np.max(nodes[:,0])-np.min(nodes[:,0])),
                     np.max(nodes[:,0])+0.01*np.abs(
                                np.max(nodes[:,0])-np.min(nodes[:,0])),
                     L)
            yc0 = np.linspace(
                     np.min(nodes[:,1])-0.01*np.abs(
                                np.max(nodes[:,1])-np.min(nodes[:,1])), 
                     np.max(nodes[:,1])+0.01*np.abs(
                                np.max(nodes[:,1])-np.min(nodes[:,1])),
                     M)
            zc0 = np.linspace(
                     np.min(nodes[:,2])-0.01*np.abs(
                            np.max(nodes[:,2])-np.min(nodes[:,2])), 
                     np.max(nodes[:,2])+0.01*np.abs(
                            np.max(nodes[:,2])-np.min(nodes[:,2])),
                     N)
            cnt_v = 0
            for i in range(L):
                for j in range(M):
                    for k in range(N):
                        V0[p, cnt_v, :] = np.array([xc0[i], yc0[j], zc0[k]])
                        cnt_v += 1
            
            # Normalized point cloud
            s = nodes_n[:,0]
            t = nodes_n[:,1]
            u = nodes_n[:,2]
        
            # Bernstein matrix
            shpBern = np.max([L, M, N])
            Mbernstein = np.matrix(np.zeros([shpBern, shpBern]))
            Mbernstein[0,0] = 1
            for i in range(1, Mbernstein.shape[0]):
                for j in range(Mbernstein.shape[1]):
                    if j == 1: Mbernstein[i,j] = 1
                    Mbernstein[i,j] = Mbernstein[i-1, j-1] + Mbernstein[i-1, j]

            # Bernstein trivariate-polynomial coefficients
            cnt_b = 0
            for i in range (L):
                s_term = Mbernstein[L-1,i]*(1-s)**(L-1-i)*s**i
                for j in range(M):
                    t_term = Mbernstein[M-1,j]*(1-t)**(M-1-j)*t**j
                    for k in range(N):
                        B[p, :, cnt_b] = s_term*t_term*(
                            Mbernstein[N-1,k]*(1-u)**(N-1-k)*u**k)
                        cnt_b += 1
        
        # Generate graphs
        pcs_t = tf.constant(pc_temp, tf.float32, name="pc_templates")
        ind_t = tf.constant(np.array(range(pc_temp.shape[0])), tf.int64,
                            name="temp_ind")

        # Output
        return(B, V0, pcs_t, ind_t)

    # Training PC-AE with MSD
    def pc_ae_training(pcae_config, GPUid=-1):
        ''' Function for training the PC-AE on CPU/GPU with MSD

        Input:
          - pcae_config: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - (FLAG_ERROR, True), if the Function was interrupted by an error
          - (FLAG_ERROR, False), if the Function finished without errors.
        '''

        ## Allocate GPU/CPU
        os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(GPUid))

        ## Read configuration dictionary
        if os.path.exists(pcae_config):
            os.system("cp {} configdict.py".format(pcae_config))
            from configdict import confignet as config
        else:
            print("ERROR! Configuration file not found!")
            print("File: {}".format(pcae_config))
            return('FLAG_ERROR', True)

        ## Create output directory
        if type(config["out_data"]) == type(None):
            out_dir = "."
        else:
            out_dir = str(config["out_data"])
        if not os.path.exists(out_dir): os.mkdir(out_dir)

        ## Create network directory
        net_dir = "{}/{}".format(out_dir, str(config["net_id"]))
        if not os.path.exists(net_dir): os.mkdir(net_dir)
        # copy dictionary to the network directory
        os.system("cp {} {}".format(pcae_config, net_dir))

        ## Load data set
        data_training, list_batches_training, data_test, list_batches_test,\
            pc_size, samples_training,\
            samples_test = arch_training.data_loader(config)
        ## Normalize the data set
        # Training
        data_training, normlimits = CAE2PC.data_set_norm(data_training, 
                                                         np.array([0.1, 0.9]))
        # Test
        data_test = CAE2PC.data_set_norm(data_test, np.array([0.1,0.9]),
                                         inp_lim=normlimits)[0]
        
        ## Save files: Normalization limits and logs with shape names
        # .npy format
        np.save("{}/norm_inp_limits".format(net_dir), normlimits)
        np.save("{}/log_files_training".format(net_dir), samples_training)
        np.save("{}/log_files_test".format(net_dir), samples_test)
        # Text (.dat) format
        with open("{}/norm_inp_limits.dat".format(net_dir), 'w') as file:
            file.write(str(normlimits[0])+","+str(normlimits[1])+"\n")
            file.close()
        with open("{}/log_files_training.dat".format(net_dir), 'w') as file:
            for line in list(samples_training):
                file.write(line+"\n")
            file.close()
        with open("{}/log_files_test.dat".format(net_dir), 'w') as file:
            for line in list(samples_test):
                file.write(line+"\n")
            file.close()
        
        ## Generate architecture
        # Archtiecture settings
        latent_layer = int(config["latent_layer"])
        encoder_sizes = list(config["encoder_layers"])
        encoder_sizes.append(latent_layer)
        decoder_sizes = list(config["decoder_layers"])
        decoder_sizes.append(pc_size)
        # Generate graph
        S_in, _, pc = PC_AE.pcae(encoder_sizes, pc_size, latent_layer,
                                decoder_sizes)[:3]

        ## Path to autoecoder graph files
        pathToGraph = "{}/pcae".format(net_dir)

        ## Training algorithm and losses
        # Settings
        l_rate = float(config["l_rate"])
        max_epochs = int(config["epochs_max"])
        autosave_rate = float(config["autosave_rate"])
        crit_stop = float(config["stop_training"])
        # Optimizer and losses
        optimizer = tf.train.AdamOptimizer(l_rate)
        msd = losses.msd_gpu(S_in, pc)
        method = optimizer.minimize(msd)
        
        ## Initialize variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        ## Initialize tensorflow session
        with tf.Session() as sess:
            # Initialize parameters
            sess.run(init)
            
            ## Iteration over epochs
            # Initial time
            t = time.time()
            # List for logging training losses
            loss_values = []
            # List for logging training losses
            test_loss = []
            # Epochs
            for epochs in range(max_epochs):
                # Temporary list for losses
                loss_temp = []
                # Iteration over training batches
                for b in range(len(list_batches_training)):
                    # Train 1 epoch over batch
                    _, loss_val  = sess.run([method, msd], 
                                    feed_dict={S_in: 
                                                data_training
                                                [list_batches_training[b],:,:]})
                    # Log loss value
                    loss_temp.append(loss_val)
                # Log mean and std over batches
                loss_values.append([np.mean(loss_temp), np.std(loss_temp)])
            
                # Iteration over test batches
                test_loss_temp = []
                for b in range(len(list_batches_test)):
                    # Test over one batch
                    test_val = sess.run(msd, feed_dict={S_in: 
                                                data_test
                                                [list_batches_test[b],:,:]})
                    # Log loss for a batch
                    test_loss_temp.append(test_val)
                # Log mean and std over batches
                test_loss.append([np.mean(test_loss_temp), 
                                  np.std(test_loss_temp)])
                
                # Generate logs, ploting and saving graph
                if (epochs+1) % autosave_rate == 0 and epochs > 0:
                    # Elapsed time
                    etime = time.time()-t
                    timeh = np.floor(etime/3600)
                    timemin = np.floor((etime/60 - timeh*60))
                    timesec = (etime - timeh*3600 - timemin*60)
                    # Remaining time
                    rtime = (max_epochs-epochs)*(etime/epochs)
                    rtimeh = np.floor(rtime/3600)
                    rtimemin = np.floor((rtime/60 - rtimeh*60))
                    rtimesec = (rtime - rtimeh*3600 - rtimemin*60)

                    # Print current and estimate runtime on screen            
                    print(str.format("EPOCH: {}", epochs+1))
                    print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
                        "h",  str.format("{:.0f}", timemin), "min", \
                            str.format("{:.0f}", timesec), "s")
                    print("REMAINING TIME:", str.format("{:.0f}", rtimeh), \
                        "h",  str.format("{:.0f}", rtimemin), "min", \
                            str.format("{:.0f}", rtimesec), "s")
            
                    # Report losses
                    print(
                        str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}",
                                   np.mean(loss_temp), np.std(loss_temp)))
                    print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}",
                          np.mean(test_loss_temp), np.std(test_loss_temp)))
                    # Save log: losses
                    pd.DataFrame(loss_values).to_csv(
                        str.format("{}/losses_training.csv", net_dir),
                        header=None, index=None)
                    pd.DataFrame(test_loss).to_csv(
                        str.format("{}/losses_test.csv", net_dir),
                        header=None, index=None)
        
                    # Save Graph
                    saver.save(sess, pathToGraph)
            
                # Finish training    
                if test_loss[-1][0] <= crit_stop: 
                    print("Stop criteria achieved!")
                    print("Test CD: {:.3E}, crit.: {:.3E}".\
                        format(test_loss[-1][0], crit_stop))
                    break
            
            ## Finish training
            # Elapsed time
            etime = time.time()-t
            timeh = np.floor(etime/3600)
            timemin = np.floor((etime/60 - timeh*60))
            timesec = (etime - timeh*3600 - timemin*60)
            # Remaining time
            rtime = (max_epochs-epochs)*(etime/epochs)
            rtimeh = np.floor(rtime/3600)
            rtimemin = np.floor((rtime/60 - rtimeh*60))
            rtimesec = (rtime - rtimeh*3600 - rtimemin*60)
    
            # Report elapsed time
            print(str.format("EPOCH: {}", epochs+1))
            print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
                "h",  str.format("{:.0f}", timemin), "min", \
                    str.format("{:.0f}", timesec), "s")
    
            # Report losses
            print(
                str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}",
                           np.mean(loss_temp), np.std(loss_temp)))
            print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}",
                  np.mean(test_loss_temp), np.std(test_loss_temp)))
            # Save log: losses
            pd.DataFrame(loss_values).to_csv(
                str.format("{}/losses_training.csv", net_dir),
                header=None, index=None)
            pd.DataFrame(test_loss).to_csv(
                str.format("{}/losses_test.csv", net_dir),
                header=None, index=None)

            # Save Graph
            saver.save(sess, pathToGraph)
            sess.close()

        # Exit message
        print("Training concluded!")

        # Save log file with elapsed and current time on network directory
        os.system("echo ELAPSED TIME: {:.0f} h {:.0f} min {:.0f} s >> {}/net_DONE".format(timeh, timemin, timesec, net_dir))
        current_time = time.strftime("%H:%M:%S", time.localtime())
        os.system("echo CURRENT TIME: {} >> {}/net_DONE".format(current_time,
                                                                net_dir))

        # Return error message
        return('FLAG_ERROR', False)

    # Training PC-VAE
    def pc_vae_training(pcvae_config, GPUid=-1):
        ''' Function for training the PC-VAE on CPU/GPU with MSD

        Input:
          - pcae_config: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - (FLAG_ERROR, True), if the Function was interrupted by an error
          - (FLAG_ERROR, False), if the Function finished without errors.
        '''

        ## Allocate GPU/CPU
        os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(GPUid))

        ## Read configuration dictionary
        if os.path.exists(pcvae_config):
            os.system("cp {} configdict.py".format(pcvae_config))
            from configdict import confignet as config
        else:
            print("ERROR! Configuration file not found!")
            print("File: {}".format(pcvae_config))
            return('FLAG_ERROR', True)

        ## Create output directory
        if type(config["out_data"]) == type(None):
            out_dir = "."
        else:
            out_dir = str(config["out_data"])

        ## Create network directory
        net_dir = "{}/{}".format(out_dir, str(config["net_id"]))
        if not os.path.exists(net_dir): os.mkdir(net_dir)
        # copy dictionary to the network directory
        os.system("cp {} {}".format(pcvae_config, net_dir))

        ## Load data set
        data_training, list_batches_training, data_test, list_batches_test,\
            pc_size, samples_training,\
            samples_test = arch_training.data_loader(config)
        ## Normalize the data set
        # Training
        data_training, normlimits = CAE2PC.data_set_norm(data_training, 
                                                         np.array([0.1, 0.9]))
        # Test
        data_test = CAE2PC.data_set_norm(data_test, np.array([0.1,0.9]),
                                         inp_lim=normlimits)[0]
        
        ## Save files: Normalization limits and logs with shape names
        # .npy format
        np.save("{}/norm_inp_limits".format(net_dir), normlimits)
        np.save("{}/log_files_training".format(net_dir), samples_training)
        np.save("{}/log_files_test".format(net_dir), samples_test)
        # Text (.dat) format
        with open("{}/norm_inp_limits.dat".format(net_dir), 'w') as file:
            file.write(str(normlimits)+","+str(normlimits[1])+"\n")
            file.close()
        with open("{}/log_files_training.dat".format(net_dir), 'w') as file:
            for line in list(samples_training):
                file.write(line[0]+"\n")
            file.close()
        with open("{}/log_files_test.dat".format(net_dir), 'w') as file:
            for line in list(samples_test):
                file.write(line+"\n")
            file.close()

        ## Generate architecture
        latent_layer = int(config["latent_layer"])
        encoder_sizes = list(config["encoder_layers"])
        encoder_sizes.append(latent_layer)
        decoder_sizes = list(config["decoder_layers"])
        decoder_sizes.append(pc_size)
        S_in, _,S_out,mu,sgm,kdout, = PC_VAE.pcvae(encoder_sizes, 
                                                 pc_size, latent_layer,
                                                decoder_sizes)[:6]

        ## Path to autoecoder graph files
        pathToGraph = "{}/pcvae".format(net_dir)

        ## Training algorithm and losses
        # Settings
        l_rate = float(config["l_rate"])
        alpha1 = float(config["alpha1"])
        alpha2 = float(config["alpha2"])
        dpout = float(config["dpout"])
        max_epochs = int(config["epochs_max"])
        autosave_rate = float(config["autosave_rate"])
        crit_stop = float(config["stop_training"])
        # Optimizer and losses
        optimizer = tf.train.AdamOptimizer(l_rate)
        rec_loss = tf.math.scalar_mul(alpha1,losses.msd_gpu(S_in, S_out))
        kld_loss = tf.math.scalar_mul(alpha2,losses.KLD(sgm, mu))
        loss_function = rec_loss + kld_loss
        method = optimizer.minimize(loss_function)
        
        ## Initialize variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        ## Initialize session
        with tf.Session() as sess:
            # Initialize parameters
            sess.run(init)
            
            ## Iteration over epochs
            # Initial time
            t = time.time()
            # List for logging training losses
            loss_values = []
            # List for logging training losses
            test_loss = []
            # Epochs
            for epochs in range(max_epochs):
                # temporary list for log
                loss_temp = []
                # Iteration over training batches
                for b in range(len(list_batches_training)):
                    # Train 1 epoch over batch
                    _, loss_val  = sess.run([method, loss_function], 
                                    feed_dict={S_in: 
                                                data_training
                                                [list_batches_training[b],:,:],
                                               kdout: dpout})
                    # Log loss value
                    loss_temp.append(loss_val)
                # Log mean and std over batches
                loss_values.append([np.mean(loss_temp), np.std(loss_temp)])
            
                # Iteration over test batches
                test_loss_temp = []
                for b in range(len(list_batches_test)):
                    # Test over one batch
                    test_val = sess.run(loss_function, 
                                        feed_dict={S_in: 
                                                   data_test
                                                   [list_batches_test[b],:,:],
                                                   kdout: 1.0})
                    # Log loss for a batch
                    test_loss_temp.append(test_val)
                # Log mean and std over batches
                test_loss.append([np.mean(test_loss_temp), 
                                  np.std(test_loss_temp)])
                
                # Generate logs, ploting and saving graph
                if (epochs+1) % autosave_rate == 0 and epochs > 0:
                    # Elapsed time
                    etime = time.time()-t
                    timeh = np.floor(etime/3600)
                    timemin = np.floor((etime/60 - timeh*60))
                    timesec = (etime - timeh*3600 - timemin*60)
                    # Remaining time
                    rtime = (max_epochs-epochs)*(etime/epochs)
                    rtimeh = np.floor(rtime/3600)
                    rtimemin = np.floor((rtime/60 - rtimeh*60))
                    rtimesec = (rtime - rtimeh*3600 - rtimemin*60)

                    # Print current and estimate runtime on screen            
                    print(str.format("EPOCH: {}", epochs+1))
                    print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
                        "h",  str.format("{:.0f}", timemin), "min", \
                            str.format("{:.0f}", timesec), "s")
                    print("REMAINING TIME:", str.format("{:.0f}", rtimeh), \
                        "h",  str.format("{:.0f}", rtimemin), "min", \
                            str.format("{:.0f}", rtimesec), "s")
            
                    # Report losses
                    print(
                        str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}",
                                   np.mean(loss_temp), np.std(loss_temp)))
                    print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}",
                          np.mean(test_loss_temp), np.std(test_loss_temp)))
                    # Save log: losses
                    pd.DataFrame(loss_values).to_csv(
                        str.format("{}/losses_training.csv", net_dir),
                        header=None, index=None)
                    pd.DataFrame(test_loss).to_csv(
                        str.format("{}/losses_test.csv", net_dir),
                        header=None, index=None)
        
                    # Save Graph
                    saver.save(sess, pathToGraph)
            
                # Finish training    
                if test_loss[-1][0] <= crit_stop: 
                    print("Stop criteria achieved!")
                    print("Test CD: {:.3E}, crit.: {:.3E}".\
                        format(test_loss[-1][0], crit_stop))
                    break

            # Finish training
            # Elapsed time
            etime = time.time()-t
            timeh = np.floor(etime/3600)
            timemin = np.floor((etime/60 - timeh*60))
            timesec = (etime - timeh*3600 - timemin*60)
            # Remaining time
            rtime = (max_epochs-epochs)*(etime/epochs)
            rtimeh = np.floor(rtime/3600)
            rtimemin = np.floor((rtime/60 - rtimeh*60))
            rtimesec = (rtime - rtimeh*3600 - rtimemin*60)
    
            print(str.format("EPOCH: {}", epochs+1))
            print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
                "h",  str.format("{:.0f}", timemin), "min", \
                    str.format("{:.0f}", timesec), "s")
    
            # Report losses
            print(
                str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}",
                           np.mean(loss_temp), np.std(loss_temp)))
            print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}",
                  np.mean(test_loss_temp), np.std(test_loss_temp)))
            # Save log: losses
            pd.DataFrame(loss_values).to_csv(
                str.format("{}/losses_training.csv", net_dir),
                header=None, index=None)
            pd.DataFrame(test_loss).to_csv(
                str.format("{}/losses_test.csv", net_dir),
                header=None, index=None)

            # Save Graph
            saver.save(sess, pathToGraph)
            sess.close()

        # Exit message
        print("Training concluded")

        # Save log file with elapsed and current time on network directory
        os.system("echo ELAPSED TIME: {:.0f} h {:.0f} min {:.0f} s >> {}/net_DONE".format(timeh, timemin, timesec, net_dir))
        current_time = time.strftime("%H:%M:%S", time.localtime())
        os.system("echo CURRENT TIME: {} >> {}/net_DONE".format(current_time,
                                                                net_dir))

        # Return error message
        return('FLAG_ERROR', False)
    
    # Training Point2FFD
    def point2ffd_training(p2ffd_config, GPUid=-1):
        ''' Function for training the PC-AE on CPU/GPU with MSD

        Input:
          - pcae_config: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - (FLAG_ERROR, True), if the Function was interrupted by an error
          - (FLAG_ERROR, False), if the Function finished without errors.
        '''

        ## Allocate GPU/CPU
        os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(GPUid))

        ## Read configuration dictionary
        if os.path.exists(p2ffd_config):
            os.system("cp {} configdict.py".format(p2ffd_config))
            from configdict import confignet as config
        else:
            print("ERROR! Configuration file not found!")
            print("File: {}".format(p2ffd_config))
            return('FLAG_ERROR', True)

        ## Create output directory
        if type(config["out_data"]) == type(None):
            out_dir = "."
        else:
            out_dir = str(config["out_data"])
        if not os.path.exists(out_dir): os.mkdir(out_dir)

        ## Create network directory
        net_dir = "{}/{}".format(out_dir, str(config["net_id"]))
        if not os.path.exists(net_dir): os.mkdir(net_dir)
        # copy dictionary to the network directory
        os.system("cp {} {}".format(p2ffd_config, net_dir))

        ## Load data set
        data_training, list_batches_training, data_test, list_batches_test,\
            pc_size, samples_training,\
            samples_test = arch_training.data_loader(config)
        ## Normalize the data set
        # Training
        data_training, normlimits = CAE2PC.data_set_norm(data_training, 
                                                         np.array([0.1, 0.9]))
        # Test
        data_test = CAE2PC.data_set_norm(data_test, np.array([0.1,0.9]),
                                         inp_lim=normlimits)[0]

        ## Save files: Normalization limits and logs with shape names
        # .npy format
        np.save("{}/norm_inp_limits".format(net_dir), normlimits)
        np.save("{}/log_files_training".format(net_dir), samples_training)
        np.save("{}/log_files_test".format(net_dir), samples_test)
        # Text (.dat) format
        with open("{}/norm_inp_limits.dat".format(net_dir), 'w') as file:
            file.write(str(normlimits[0])+","+str(normlimits[1])+"\n")
            file.close()
        with open("{}/log_files_training.dat".format(net_dir), 'w') as file:
            for line in list(samples_training):
                file.write(line+"\n")
            file.close()
        with open("{}/log_files_test.dat".format(net_dir), 'w') as file:
            for line in list(samples_test):
                file.write(line+"\n")
            file.close()

        ## Load templates
        B, V0, pc_temp, ind_t = arch_training.meshtemp_loader(config, pc_size,
                                                              normlimits)
        
        ## Generate architecture
        # Archtiecture settings
        # Encoder 
        latent_layer = int(config["latent_layer"])
        encoder_sizes = list(config["encoder_layers"])
        encoder_sizes.append(latent_layer)
        sigma_n = float(config['sigma_n'])
        noise_flag = float(config['gamma_n'])
        # Classifier
        class_sizes = config["class_layers"]
        class_sizes.append(V0.shape[0])
        # Decoder
        decoder_sizes = list(config["decoder_layers"])
        decoder_sizes.append(V0.shape[1])

        # Generate graph
        S_in, Z, S_out, class_labl, class_prob, gamma_noise = \
            Point2FFD.point2ffd(B, V0,
                                encoder_sizes, pc_size, 
                                latent_layer, class_sizes,
                                decoder_sizes, sigma_n=sigma_n)

        ## Path to autoecoder graph files
        pathToGraph = "{}/p2ffd".format(net_dir)

        ## Training algorithm and losses
        # Settings
        l_rate = float(config["l_rate"])
        max_epochs = int(config["epochs_max"])
        autosave_rate = float(config["autosave_rate"])
        crit_stop = float(config["stop_training"])
        # Optimizer and losses
        optimizer = tf.train.AdamOptimizer(l_rate)
        loss_comb = losses.msd_comb(S_in, S_out, pc_temp, ind_t, class_prob)
        method = optimizer.minimize(loss_comb)

        ## Initialize variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        ## Initialize tensorflow session
        with tf.Session() as sess:
            # Initialize parameters
            sess.run(init)
            
            ## Iteration over epochs
            # Initial time
            t = time.time()
            # List for logging training losses
            loss_values = []
            # List for logging training losses
            test_loss = []
            # Epochs
            for epochs in range(max_epochs):
                # Temporary list for losses
                loss_temp = []
                # Iteration over training batches
                for b in range(len(list_batches_training)):
                    # Train 1 epoch over batch
                    _, loss_val  = sess.run([method, loss_comb], 
                                    feed_dict={S_in: 
                                                data_training
                                                [list_batches_training[b],:,:],
                                               gamma_noise: noise_flag})
                    # Log loss value
                    loss_temp.append(loss_val)
                # Log mean and std over batches
                loss_values.append([np.mean(loss_temp), np.std(loss_temp)])
            
                # Iteration over test batches
                test_loss_temp = []
                for b in range(len(list_batches_test)):
                    # Test over one batch
                    test_val = sess.run(loss_comb, feed_dict={S_in: 
                                                data_test
                                                [list_batches_test[b],:,:],
                                               gamma_noise: 0})
                    # Log loss for a batch
                    test_loss_temp.append(test_val)
                # Log mean and std over batches
                test_loss.append([np.mean(test_loss_temp), 
                                  np.std(test_loss_temp)])
                
                # Generate logs, ploting and saving graph
                if (epochs+1) % autosave_rate == 0 and epochs > 0:
                    # Elapsed time
                    etime = time.time()-t
                    timeh = np.floor(etime/3600)
                    timemin = np.floor((etime/60 - timeh*60))
                    timesec = (etime - timeh*3600 - timemin*60)
                    # Remaining time
                    rtime = (max_epochs-epochs)*(etime/epochs)
                    rtimeh = np.floor(rtime/3600)
                    rtimemin = np.floor((rtime/60 - rtimeh*60))
                    rtimesec = (rtime - rtimeh*3600 - rtimemin*60)

                    # Print current and estimate runtime on screen            
                    print(str.format("EPOCH: {}", epochs+1))
                    print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
                        "h",  str.format("{:.0f}", timemin), "min", \
                            str.format("{:.0f}", timesec), "s")
                    print("REMAINING TIME:", str.format("{:.0f}", rtimeh), \
                        "h",  str.format("{:.0f}", rtimemin), "min", \
                            str.format("{:.0f}", rtimesec), "s")
            
                    # Report losses
                    print(
                        str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}",
                                   np.mean(loss_temp), np.std(loss_temp)))
                    print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}",
                          np.mean(test_loss_temp), np.std(test_loss_temp)))
                    # Save log: losses
                    pd.DataFrame(loss_values).to_csv(
                        str.format("{}/losses_training.csv", net_dir),
                        header=None, index=None)
                    pd.DataFrame(test_loss).to_csv(
                        str.format("{}/losses_test.csv", net_dir),
                        header=None, index=None)
        
                    # Save Graph
                    saver.save(sess, pathToGraph)
            
                # Finish training    
                if test_loss[-1][0] <= crit_stop: 
                    print("Stop criteria achieved!")
                    print("Test CD: {:.3E}, crit.: {:.3E}".\
                        format(test_loss[-1][0], crit_stop))
                    break
            
            ## Finish training
            # Elapsed time
            etime = time.time()-t
            timeh = np.floor(etime/3600)
            timemin = np.floor((etime/60 - timeh*60))
            timesec = (etime - timeh*3600 - timemin*60)
            # Remaining time
            rtime = (max_epochs-epochs)*(etime/epochs)
            rtimeh = np.floor(rtime/3600)
            rtimemin = np.floor((rtime/60 - rtimeh*60))
            rtimesec = (rtime - rtimeh*3600 - rtimemin*60)
    
            # Report elapsed time
            print(str.format("EPOCH: {}", epochs+1))
            print("ELAPSED TIME:", str.format("{:.0f}", timeh), \
                "h",  str.format("{:.0f}", timemin), "min", \
                    str.format("{:.0f}", timesec), "s")
    
            # Report losses
            print(
                str.format("\t- Training :: Losses = {:.3E} +/- {:.3E}",
                           np.mean(loss_temp), np.std(loss_temp)))
            print(str.format("\t- Test :: Losses = {:.3E} +/- {:.3E}",
                  np.mean(test_loss_temp), np.std(test_loss_temp)))
            # Save log: losses
            pd.DataFrame(loss_values).to_csv(
                str.format("{}/losses_training.csv", net_dir),
                header=None, index=None)
            pd.DataFrame(test_loss).to_csv(
                str.format("{}/losses_test.csv", net_dir),
                header=None, index=None)

            # Save Graph
            saver.save(sess, pathToGraph)
            sess.close()

        # Exit message
        print("Training concluded!")

        # Save log file with elapsed and current time on network directory
        os.system("echo ELAPSED TIME: {:.0f} h {:.0f} min {:.0f} s >> {}/net_DONE".format(timeh, timemin, timesec, net_dir))
        current_time = time.strftime("%H:%M:%S", time.localtime())
        os.system("echo CURRENT TIME: {} >> {}/net_DONE".format(current_time,
                                                                net_dir))

        # Return error message
        return('FLAG_ERROR', False)

    # Reconstruction losses
    def reconstruction_losses(config_path, GPUid=-1):
        '''Function to calculate the reconstruction losses (Chamfer Distance)
        on the data set (training + testing)

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - Log file with the evaluated metrics is stored directly on the
          network directory at "network_verification/network_verification.dat"
        '''

        ## Allocate GPU/CPU
        os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(GPUid))
        
        ## Read configuration dictionary
        if os.path.exists(config_path):
            os.system("cp {} configdict.py".format(config_path))
            from configdict import confignet as config
        else:
            print("ERROR! Configuration file not found!")
            print("File: {}".format(config_path))
            return()
        
       ## Output directory
        if type(config["out_data"]) == type(None):
            out_dir = "."
        else:
            out_dir = str(config["out_data"])
        ## Network directory
        net_dir = "{}/{}".format(out_dir, str(config["net_id"]))
        
        ## Load data set
        data_training, _, data_test, _, _, samples_training,\
            samples_test = arch_training.data_loader(config)
        
        ## Normalize the data set
        # Load normalization limits
        normlim = np.load("{}/norm_inp_limits.npy".format(net_dir))
        # Training
        data_training = CAE2PC.data_set_norm(data_training,
                                            np.array([0.1, 0.9]),
                                            inp_lim=normlim)[0]
        # Test
        data_test = CAE2PC.data_set_norm(data_test, np.array([0.1,0.9]),
                                         inp_lim=normlim)[0]
        ## Concatenate data
        # Point Clouds
        nshapes = data_training.shape[0] + data_test.shape[0]
        data_set_shape = [nshapes, data_test.shape[1], data_test.shape[2]]
        data_set = np.zeros(data_set_shape)
        data_set[0:data_training.shape[0],:,:] = data_training
        data_set[data_training.shape[0]:, :, :] = data_test
        # File names
        list_names = list(samples_training) + list(samples_test)
        # Load the names of the files utilized during training
        list_test = list(np.load("{}/log_files_test.npy".format(net_dir)))

        ## Load the architecture
        flag_vae = False
        flag_p2ffd = False
        # In case the network is PC-AE
        try:
            # - Import Graph at latest state (after training)
            TFmetaFile = str.format("{}/pcae.meta", net_dir)
            TFDirectory = str.format("{}/", net_dir)
            # import graph data
            new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                   clear_devices=True)
        # In case the network is PC-VAE
        except:
            try:
                # - Import Graph at latest state (after training)
                TFmetaFile = str.format("{}/pcvae.meta", net_dir)
                TFDirectory = str.format("{}/", net_dir)
                # import graph data
                new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                       clear_devices=True)
                flag_vae = True
            except:
                # - Import Graph at latest state (after training)
                TFmetaFile = str.format("{}/p2ffd.meta", net_dir)
                TFDirectory = str.format("{}/", net_dir)
                # import graph data
                new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                       clear_devices=True)
                flag_p2ffd = True


        ## Create directory to save the files
        output_test = str.format("{}/network_verification", net_dir)
        if not os.path.exists(output_test):
            os.system(str.format("mkdir {}", output_test))

        ## Evaluation of the shapes
        with tf.Session() as sess:
            ## Restore last state of the graph
            new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
            graph = tf.get_default_graph()

            ## Import network layers
            # - Input
            x = graph.get_tensor_by_name("S_in:0")
            # - Latent Representation
            Z = graph.get_tensor_by_name("Z:0")
            Z_size = Z.shape[1]
            # - Point clouds
            S_out = graph.get_tensor_by_name("S_out:0")
            # Droput (PC-VAE)
            if flag_vae: dpout = graph.get_tensor_by_name("do_rate:0")
            # Gamma parameter for Gaussian noise (Point2FFD)
            if flag_p2ffd: gamma_n = graph.get_tensor_by_name("gamma_n:0")

            ## Calculate reconstruction losses and latent representations
            # Data log
            eval_log = np.zeros((data_set.shape[0], 1+Z_size+2), dtype="U200")
            print("\n")
            # Loop over loaded shapes
            for i in range(nshapes):
                print('Shape {} of {}'.format(i+1, nshapes), end="\r")
                xin = np.reshape(data_set[i,:,:], (1, -1, 3))
                # Calculate the latent representation and point cloud 
                # reconstruction
                if flag_vae:
                    lr, xrec = sess.run([Z, S_out],\
                        feed_dict={x: xin, dpout: 1.0})
                else:
                    if flag_p2ffd:
                        lr, xrec = sess.run([Z, S_out], feed_dict={x: xin,
                                                                   gamma_n:0})
                    else:
                        lr, xrec = sess.run([Z, S_out], feed_dict={x: xin})
                
                # Calculate Chamfer Distance
                CD = losses.CD(xin[0,:,:], xrec[0,:,:])

                # Generate log
                eval_log[i,0] = list_names[i].split("/")[-1].split(".")[0]
                eval_log[i, 1:1+Z_size] = lr.flatten()
                eval_log[i, 1+Z_size:] = CD
                eval_log[i, 2+Z_size:] = int(list_names[i] in \
                                                             list_test)
            sess.close()
        
        # Write file with results
        header_f = ["shape_id",]
        for i in range(Z_size): header_f += ["z_{}".format(i)]
        header_f += ["chamfer_distance", "test_set"]
        pd.DataFrame(eval_log).to_csv("{}/network_verification.dat".\
            format(output_test),\
            header=header_f, index=None)
        
        # End of the function
        return()