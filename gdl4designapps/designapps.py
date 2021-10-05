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

Authors: Thiago Rios, Sneha Saha
Contact: gdl4designapps@honda-ri.de
"""

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------
# Basic tools
import numpy as np
import os
import ast
#from pandas._config import config

# Visualization tools
import pyvista as pv
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
import tensorflow as tf

# GDL4DesignApps
from gdl4designapps.preprocess_methods import CAE2PC

# ------------------------------------------------------------------------------
# VISUALIZATION TOOLS
# ------------------------------------------------------------------------------

# (Geometric) Data visualization
class Vis3D:
    # Plot 3D point clouds
    def pcplot(Splot, figname, pointsize=20, colorpoints='tan'):
        '''
        '''
        pltv = pv.Plotter(off_screen=True, window_size=(3440, 1440))
        pltv.set_background("white")
        pltv.add_mesh(Splot, point_size=pointsize, style="points",
                      render_points_as_spheres=True,
                      lighting=True, color=colorpoints)
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        pltv.view_isometric()
        pltv.show(screenshot=figname)
        pltv.close()
        return()
    
    # Plot color maps onto 3D point clouds
    def pccmap(Splot, cscalar, figname, pointsize=20, vlim=[-1,1]):
        '''
        '''
        pltv = pv.Plotter(off_screen=True, window_size=(3440, 1440))
        pv.set_plot_theme("document")
        pltv.set_background("white")
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        pltv.add_mesh(Splot, point_size=pointsize, style="points",
                      render_points_as_spheres=True, 
                      lighting=True, scalars=cscalar.flatten(), cmap=cm_custm,
                      show_scalar_bar=True, clim=vlim)
        pltv.add_scalar_bar(title_font_size=48, label_font_size=40,
                            shadow=True, n_labels=4,
                            color='black', italic=True, fmt="%.1E",
                            font_family="times")
        pltv.view_isometric()
        pltv.show(screenshot=figname)
        pltv.close()
        return()

    # Comparison between point clouds
    def pccomp(Sref, Scomp, figname, pointsize=20, cdata=None, vlim=[0,1]):
        '''
        '''
        pltv = pv.Plotter(off_screen=True, 
                          window_size=(3440, 1440))
        pv.set_plot_theme("document")
        pltv.set_background("white")
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        
        pltv.add_mesh(Sref, point_size=pointsize, style="points",
                      render_points_as_spheres=True,
                      lighting=True, color="#00334c")
        Sref_yspan = np.max(Sref[:,1]) - np.min(Sref[:,1])
        if not type(cdata)==type(None):
            pltv.add_mesh(Scomp+[0,2*Sref_yspan,0], 
                          point_size=pointsize, style="points", 
                          render_points_as_spheres=True,
                          lighting=True, scalars=cdata.flatten(), cmap=cm_custm,
                          show_scalar_bar=True, clim=vlim)
            pltv.add_scalar_bar(title_font_size=48, label_font_size=40,
                            shadow=True, n_labels=4,
                            color='black', italic=True, fmt="%.1E",
                            font_family="times")
        else:
            pltv.add_mesh(Scomp+[0,2*Sref_yspan,0], 
                      point_size=pointsize, style="points",
                      render_points_as_spheres=True,
                      lighting=True, color="#c8102e")
        
        pltv.view_isometric()
        pltv.show(screenshot=figname)
        pltv.close()
        return()

        d=0

    # Plot meshes
    def msplot(mesh_path, figname, cmesh='tan'):
        '''
        '''
        pltv = pv.Plotter(off_screen=True, window_size=(3440, 1440))
        pltv.set_background("white")
        pltv.add_mesh(pv.PolyData(mesh_path), lighting=True, show_edges=True, 
                         color=cmesh)
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        pltv.view_isometric()
        pltv.show(screenshot=figname)
        pltv.close()
        return()
    
    # Plot color maps onto meshes
    def mscmap():
        ''' 
        '''
    
    # Losses histogram
    def hist_losses(dframe, fname, fsize=(4,3)):
        _, axs = plt.subplots(figsize=fsize, dpi=300)
        f=sns.histplot(dframe, x="chamfer_distance",
                    hue="test_set",
                    kde=True,
                    stat='percent', multiple="dodge", shrink=.8,
                    bins=20,
                    palette=sns.color_palette(list(np.array(palette)[[0,2]]), 2)
                    )
        plt.tight_layout()
        axs.legend(["Test set", "Training set"], handlelength=1.0,
                    handletextpad=0.2, loc="center left", 
                    bbox_to_anchor=(0.675, 0.9), fontsize=10)
        plt.xlabel("Camfer Distance")
        plt.ylabel("Percentage")
        if fname[-3:] == "pdf":
            plt.savefig(fname, dpi=300)
        else:
            plt.savefig(fname+".pdf", dpi=300)
        plt.close()
        return()

# ------------------------------------------------------------------------------
# DESIGN APPLICATIONS
# ------------------------------------------------------------------------------

# Algorithms for generating latent representations, 3D point clouds and 
# visualizing network features
class DesignApps:
    # Calculate the latent representations of 3D point clouds
    def pointcloud_to_Z(config_path, pc_batch, GPUid=-1):
        '''Function to compress 3D point clouds to latent representations
        using a trained archtiecture.

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - z_batch: Array with the corresponding batch of latent 
          representations calculated for the input pc_batch.
          Type: <array (-1,Lz,1)>
        '''

        ## Allocate GPU
        os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(GPUid))
        
        ## Read configuration dictionary
        if os.path.exists(config_path):
            os.system("cp {} configdict.py".format(config_path))
            from configdict import confignet as config
        else:
            print("ERROR! Configuration file not found!")
            print("File: {}".format(config_path))
            return()
        
        ## Network output directory
        if type(config["out_data"]) == type(None):
            out_dir = "."
        else:
            out_dir = str(config["out_data"])
        # Network directory
        net_dir = "{}/{}".format(out_dir, str(config["net_id"]))
        
        ## Load and normalize the data set
        # check input shape
        if len(pc_batch.shape) == 2:
            pc_batch = np.reshape(pc_batch, (1,-1,3))
        # Load normalization limits
        normlim = np.load("{}/norm_inp_limits.npy".format(net_dir))
        data_set = CAE2PC.data_set_norm(pc_batch,
                                            np.array([0.1, 0.9]),
                                            inp_lim=normlim)[0]
        # Number of shapes
        nshapes = data_set.shape[0]

        ## Load the architecture
        flag_vae = False
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
            flag_vae = True
            # - Import Graph at latest state (after training)
            TFmetaFile = str.format("{}/pcvae.meta", net_dir)
            TFDirectory = str.format("{}/", net_dir)
            # import graph data
            new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                   clear_devices=True)

        ## Create directory to save the files
        output_test = str.format("{}/network_verification", net_dir)
        if not os.path.exists(output_test):
            os.system(str.format("mkdir {}", output_test))

        ## Evaluation of the shapes
        with tf.Session() as sess:
            new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
            graph = tf.get_default_graph()
            # Import network layers
            # - Input
            x = graph.get_tensor_by_name("S_in:0")
            # - Latent Representation
            Z = graph.get_tensor_by_name("Z:0")
            Z_size = Z.shape[1]
            # Droput (PC-VAE)
            if flag_vae: dpout = graph.get_tensor_by_name("do_rate:0")

            ## Calculate reconstruction losses and latent representations
            print("\n")
            # Array of latent representations
            z_batch = np.zeros((data_set.shape[0], Z_size, 1))
            for i in range(nshapes):
                print('Shape {} of {}'.format(i+1, nshapes), end="\r")
                xin = np.reshape(data_set[i,:,:], (1, -1, 3))
                # Calculate the latent representation and point cloud 
                # reconstruction
                if flag_vae:
                    z_batch[i,:,:] = sess.run(Z,\
                        feed_dict={x: xin, dpout: 1.0})[0,:,:]
                else:
                    z_batch[i,:,:] = sess.run(Z, feed_dict={x: xin})[0,:,:]
            sess.close()

        return(z_batch)
    
    # Generate 3D point clouds from latent representations
    def Z_to_pointcloud(config_path, z_batch, GPUid=-1):
        '''Function to generate 3D point clouds from a batch of latent 
        representations.

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - z_batch: Array with the corresponding batch of latent 
          representations calculated for the input pc_batch.
          Type: <array (-1,Lz,1)>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
        '''

        ## Allocate GPU
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
        
        ## Load and normalize the data set
        # check input shape
        if len(z_batch.shape) == 2:
            if z_batch.shape[0] == 1:
                pc_batch = np.reshape(z_batch, (1,-1,1))
            else:
                pc_batch = np.reshape(z_batch, (z_batch.shape[0],-1,1))
        
        # Number of shapes
        nshapes = z_batch.shape[0]

        ## Load the architecture
        flag_vae = False
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
            flag_vae = True
            # - Import Graph at latest state (after training)
            TFmetaFile = str.format("{}/pcvae.meta", net_dir)
            TFDirectory = str.format("{}/", net_dir)
            # import graph data
            new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                   clear_devices=True)

        ## Create directory to save the files
        output_test = str.format("{}/network_verification", net_dir)
        if not os.path.exists(output_test):
            os.system(str.format("mkdir {}", output_test))

        ## Evaluation of the shapes
        with tf.Session() as sess:
            new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
            graph = tf.get_default_graph()
            # Import network layers
            # - Latent representation
            Z = graph.get_tensor_by_name("Z:0")
            S_out = graph.get_tensor_by_name("S_out:0")
            pc_size = S_out.shape[1]
            # Droput (PC-VAE)
            if flag_vae: dpout = graph.get_tensor_by_name("do_rate:0")

            ## Calculate reconstruction losses and latent representations
            print("\n")
            pc_batch = np.zeros((z_batch.shape[0], pc_size, 3))
            for i in range(nshapes):
                print('Shape {} of {}'.format(i+1, nshapes), end="\r")
                zin = np.reshape(z_batch[i,:,:], (1, -1, 1))
                if flag_vae:
                    pc_batch[i,:,:] = sess.run(S_out,\
                        feed_dict={Z: zin, dpout: 1.0})[0,:,:]
                else:
                    pc_batch[i,:,:] = sess.run(S_out, feed_dict={Z: zin})[0,:,:]
            sess.close()

        # Load normalization limits
        normlim = np.load("{}/norm_inp_limits.npy".format(net_dir))
        pc_batch = CAE2PC.data_set_norm(pc_batch,
                                             inp_lim=np.array([0.1, 0.9]),
                                            out_lim=normlim)[0]
        # Output shapes
        return(pc_batch)

    # 3D Point cloud reconstruction
    def pointcloud_reconstruct(config_path, pc_batch, GPUid=-1):
        '''Function to compress 3D point clouds to latent representations
        using a trained archtiecture.

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - pc_rec: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
        '''
        # Calculate the latent representations
        z_batch = DesignApps.pointcloud_to_Z(config_path, pc_batch, GPUid=GPUid)
        # Generate 3D point clouds based on the calculated representations
        pc_rec = DesignApps.Z_to_pointcloud(config_path, z_batch, GPUid=-1)
        
        # Return reconstructed point clouds
        return(pc_rec)

    # Feature Visualization
    def featvis(config_path, pc_batch, GPUid=-1, plot=True):
        ''' Function for visualizing the features learned by the PC-AE in the
        last convolutional layer (prior to max-pooling).

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.
          - plot (default=True): Option that enables plot and sotre the
          visualizations as .png images. Type: Boolean

        Output:
          - feat_set: Set of feature values (activations) corresponding to the
          processed point clouds
          - The visualization of the features is stored as .png images in the
          point cloud directory
        '''

        ## Allocate GPU
        os.putenv('CUDA_VISIBLE_DEVICES','{}'.format(GPUid))
        ast.literal_eval
        
        ## Read configuration dictionary
        if os.path.exists(config_path):
            os.system("cp {} configdict.py".format(config_path))
            from configdict import confignet as config
        else:
            print("ERROR! Configuration file not found!")
            print("File: {}".format(config_path))
            return()
        
        ## Network output directory
        if type(config["out_data"]) == type(None):
            out_dir = "."
        else:
            out_dir = str(config["out_data"])
        # Network directory
        net_dir = "{}/{}".format(out_dir, str(config["net_id"]))

        ## Create directory to save the files
        output_test = str.format("{}/network_verification", net_dir)
        if not os.path.exists(output_test):
            os.system(str.format("mkdir {}", output_test))
        
        ## Load and normalize the data set
        # check input shape
        if len(pc_batch.shape) == 2:
            pc_batch = np.reshape(pc_batch, (1,-1,3))
        # Load normalization limits
        normlim = np.load("{}/norm_inp_limits.npy".format(net_dir))
        data_set = CAE2PC.data_set_norm(pc_batch,
                                            np.array([0.1, 0.9]),
                                            inp_lim=normlim)[0]
        # Number of shapes
        nshapes = data_set.shape[0]

        ## Load the architecture
        flag_vae = False
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
            flag_vae = True
            # - Import Graph at latest state (after training)
            TFmetaFile = str.format("{}/pcvae.meta", net_dir)
            TFDirectory = str.format("{}/", net_dir)
            # import graph data
            new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                   clear_devices=True)

        ## Create directory to save the files
        output_test = str.format("{}/network_verification", net_dir)
        if not os.path.exists(output_test):
            os.system(str.format("mkdir {}", output_test))

        ## Evaluation of the shapes
        with tf.Session() as sess:
            new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
            graph = tf.get_default_graph()
            # Import network layers
            # - Input
            x = graph.get_tensor_by_name("S_in:0")
            # - Last convolutional layer (before max pooling)
            layer_index = len(list(config["encoder_layers"]))

            feat_layer = graph.get_tensor_by_name(
                                     str.format("enclayer_{}:0", layer_index))
            # Droput (PC-VAE)
            if flag_vae: dpout = graph.get_tensor_by_name("do_rate:0")

            ## Calculate reconstruction losses and latent representations
            print("\n")
            # Array of latent representations
            feat = np.zeros((data_set.shape[0], feat_layer.shape[1], 
                                                    feat_layer.shape[2]))
            for i in range(nshapes):
                print('Shape {} of {}'.format(i+1, nshapes), end="\r")
                xin = np.reshape(data_set[i,:,:], (1, -1, 3))
                # Calculate the latent representation and point cloud 
                # reconstruction
                if flag_vae:
                    feat[i,:,:] = sess.run(feat_layer,\
                        feed_dict={x: xin, dpout: 1.0})[0,:,:]
                else:
                    feat[i,:,:] = sess.run(feat_layer, 
                                                 feed_dict={x: xin})[0,:,:]

                if plot:
                    for j in range(feat.shape[2]):
                        filename = "{}/pc_{:03}_feat{:03}.png".format(
                                                          output_test, i, j)
                        limf = [np.min(feat[i,:,j]), np.max(feat[i,:,j])]
                        Vis3D.pccmap(data_set[i,:,:], feat[i,:,j], 
                                 filename, vlim=limf)
            sess.close()
        return(feat)
