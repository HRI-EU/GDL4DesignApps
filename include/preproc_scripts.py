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

## Script with algorithms for processing the point cloud data when training
or testing the point cloud autoencoders.

Pre-requisites:
 - Python      3.6.10
 - numpy       1.19.1
 - Ubuntu      18.04
 - pandas      1.1.0
 - networkx    2.4
 - trimesh     3.8.1

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
import scipy as sp

# Geometry manipulation tools
from stl import mesh
import trimesh

# ==============================================================================
### INITIALIZATION
## Seed for Random Number Generation
    # -- CAUTION! The seed controls how the random numbers are generated and it
    #guarantees the repeatability of the experiments
    # (generation of random shapes and initialization of the variables)
np.random.seed(seed=0)
random.seed(0)

# ==============================================================================
### Support functions
## Convert obj scenes to point clouds
def as_mesh(scene_or_mesh):
    '''Convert a possible .obj scene to a mesh and return the vertices
    Inputs:
      - scene_or_mesh: trimesh object
    Output:
      - array with mesh nodes. Type: np.array, shape [N,3]

    '''
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        #assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return np.array(mesh.vertices)

# ==============================================================================
### Algorithms for reading point cloud files
## Point cloud sampling
def pointcloud_sampling(dataset_path, n_points, n_pclouds=None, p_samp=None):
    ''' Point Cloud sampling algorithm
    Inputs:
      - dataset_path: path to the directory containing point clouds. Type: str
      - n_points: size of the point cloud. Type: int
      - n_pclouds: number of point clouds to load. Three types of input are
      possible:
          - Not specified (None): all point cloud files are loaded
          - List with file names (list, str): only the specified files are
          loaded
          - Number of files to load (int): the script reads, samples and 
          assigns the first n_pclouds files to the data set.
      - p_samp: sampling probability for each point and point cloud in the
      data set. Type: dictionary, with the name of the files as keys
    Outputs:
      - dataset_full: resampled batch, type: array, (batch_size, pc_size, 3)
      - log_names: array with the name of the loaded files
    '''

    ## Verifying if the directory exists (as it was typed)
    if os.path.exists(dataset_path):
        # Get the name of the files/directories in the provided path
        dir_names = os.listdir(dataset_path)
    else:
        print("Script interrupted. Directory not found!")
        print("Path to data set: {}".format(dataset_path))
        exit()

    # Iteration over the files/directories contained in the path
    # - Counter
    cntr = 0
            
    # Check if the number of required point clouds is less than the
    #number of files in the specified path
    # -- Number of files not specified
    if n_pclouds == None:
        n_pclouds = len(dir_names)
    else:
        # -- A number of files was defined as int
        if isinstance(n_pclouds, int):
            if n_pclouds > len(dir_names):
                print("*** WARNING: user required {} point clouds, but target\
 directory contains only {} files. n_pclouds reduced to\
 {}.".format(n_pclouds, len(dir_names), len(dir_names)))
                n_pclouds = len(dir_names)
        # -- The target files were specified in a list
        else:
            if isinstance(n_pclouds, list):
                dir_names = n_pclouds
                n_pclouds = len(dir_names)
            else:
                print("** ERROR! Script interrupted.")
                print("Input 'n_pclouds' not understood. Please check if the\
 variable was assigned as None, int or list.")
                exit()
    
    # Pre-allocate the array to assign the sampled point clouds
    dataset_full = np.zeros((n_pclouds, n_points, 3))
    # List of shape names (pre-allocated)
    log_names = list(np.array(np.zeros(n_pclouds), dtype='U70'))

    # Loop over the files
    print("## Loading geometries...")
    for i in range(n_pclouds):
        print("File {} of {}".format(i+1, n_pclouds), end='\r')
        # Path to shape i
        shape_name = str.format("{}/{}", dataset_path, dir_names[i])
        # File type
        extension = shape_name[len(shape_name)-3:None]
        
        # Check if file extension can be read by the script
        ext_allow = ["obj", "stl", "xyz", "csv"]
        if not os.path.isfile(shape_name) or not extension in ext_allow:
            continue
        
        # Sample according to the type of file
        # STL or OBJ
        if extension=="stl" or extension=="obj":
            pc_temp = as_mesh(trimesh.load_mesh(shape_name))
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
            dataset_full[i,:,:] = np.reshape(pc_temp, (1, -1, 3))
            # Update counter
            cnt = cnt + 1
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
                prob_values = p_samp[dir_names[i]]
                prob_values = prob_values / np.linalg.norm(prob_values,1)
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
    
    print("\n## DONE! ## \n")
    return(dataset_full[0:cntr,:,:], log_names[0:cntr])

# ==============================================================================
### Data partitioning tree
def data_part_tree(pc_batch, n_iter):
    ''' Data Partitioning Tree (DPT) Algorithm
    Reference: Gadelha et al. (2018), https://arxiv.org/abs/1807.03520
    ### PRE-REQUISITE: the point clouds must have a number of samples that is
    a power of 2 (e.g. 1024, 2048, 4096, ...)

    Inputs:
      - pc_batch: batch of point clouds. Type: array (-1, N, 3)
      - n_iter: number of iterations (number of times the algorithm will be
       performed on the same point cloud)
    Outputs:
      - pcb_p: batch with point clouds organized according to the partitions.
      Type array (-1, N, 3)
      - pcb_dpt: point clouds divided according to the partitions. 
      Type: array (-1, 8**n_iter, N/(8**n_iter), 3)
    '''

    ## Data check
    # Batch shape
    if len(pc_batch.shape) == 2:
        nshape = [1,] + list(pc_batch.shape)
        pc_batch = np.reshape(pc_batch, nshape)
    # Is the number of points a power of 2 ?
    logN = int(np.log2(pc_batch.shape[1]))
    if not pc_batch.shape[1] == 2**logN:
        print("***ERROR: Point cloud size is not a power of 2")
        print("Point cloud size: {}, log2(N): {}".format(
            pc_batch.shape[1], np.log2(pc_batch.shape[1])
        ))
        print("Script interrupted")
        exit()
    # Number of iterations
    if pc_batch.shape[1] < 2**(3*n_iter):
        print("**WARNING: number of iterations yields infeasible partitions")
        print("Changing number of iterations from {} to {}".format(
            n_iter, int(np.log2(pc_batch.shape[1])/3)))
        n_iter = int(np.log2(pc_batch.shape[1])/3)

    # Arrays for assigning the data
    N = pc_batch.shape[1]
    pcb_p = np.zeros(pc_batch.shape)
    pcb_dpt = np.zeros((pc_batch.shape[0], 8**n_iter, int(N/(8**n_iter)), 3))
    
    for n in range(pc_batch.shape[0]):
        # temporary point cloud array
        pc_temp = np.array(pc_batch[n,:,:])
        for k in range(n_iter):
            # Number of division operations per axis
            part = [int(2**(k*3)), int(2**(3*k+1)), int(2**(3*k+2))]

            # Partition in x
            for i in range(part[0]):
                pc_temp[int(N*i/part[0]):int(N*(i+1)/part[0]), :] = pc_temp[
                    np.argsort(
                            pc_temp[int(N*i/part[0]):int(N*(i+1)/part[0]), 0])
                        +int(N*i/part[0]), :]
    
            # Partition in y
            for i in range(part[1]):
                pc_temp[int(N*i/part[1]):int(N*(i+1)/part[1]), :] = pc_temp[
                    np.argsort(
                            pc_temp[int(N*i/part[1]):int(N*(i+1)/part[1]), 1])
                        +int(N*i/part[1]), :]
    
            # Partition in z
            for i in range(part[2]):
                pc_temp[int(N*i/part[2]):int(N*(i+1)/part[2]), :] = pc_temp[
                    np.argsort(
                            pc_temp[int(N*i/part[2]):int(N*(i+1)/part[2]), 2])
                        +int(N*i/part[2]), :]
       
        # Assign point cloud to output batch
        pcb_p[n,:,:] = pc_temp
        # Assign partitioned tree
        for i in range(8**n_iter):
            pcb_dpt[n,i,:,:] = \
                pc_temp[int(N*i/(8**n_iter)):int(N*(i+1)/(8**n_iter)), :]
    
    return(pcb_p, pcb_dpt)

# ==============================================================================
### Data set normalization
def data_set_norm(pc_batch, out_lim, inp_lim=None):
    ''' Data set normalization
    Algorithm to scale the coordinates of the point clouds to an interval of in
    terest (e.g. [0,1]^3). The aspect ratio of the geometries is preserved.
    Inputs:
      - data_set: Batch of point clouds. Type: array, (-1, pc_size, 3)
      - out_lim: Output interval. Type: array, (2,)
      - inp_lim: Input interval. If not specified, the range of the data is
      considered instead. Type: array, (2,)
    Outputs:
      - pc_norm: Normalized data set. Type: array (-1, pc_size, 3)
      - inp_lim: data interval of the original data set. Type: array, (2,)
    '''

    # Assign input limits
    if np.array(inp_lim).any()==None:
        inp_lim = np.array([np.min(pc_batch), np.max(pc_batch)])

    ## Normalize the data
    pc_norm = (pc_batch - np.min(inp_lim))/(np.max(inp_lim)-np.min(inp_lim))
    pc_norm = pc_norm*(np.max(out_lim)-np.min(out_lim)) + np.min(out_lim)

    return(pc_norm, inp_lim)

# EOF
