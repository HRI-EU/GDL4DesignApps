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
# from gdl4designapps.preprocess_methods import CAE2PC
from gdl4designapps.preprocess_methods import CAE2PC

# ------------------------------------------------------------------------------
# VISUALIZATION TOOLS
# ------------------------------------------------------------------------------

# (Geometric) Data visualization
class Vis3D:
    # Plot 3D point clouds
    def pcplot(Splot, figname, pointsize=20, colorpoints="#c8102e", 
               cam_az=0, cam_el=0, wsize=(3440, 1440)):
        ''' Function for visualizing 3D point clouds.

        Input:
          - Splot: Array with the Cartesian coordinates of the 3D point cloud.
          Type: <array, (N,3)>
          - figname: Name of the file to save the figure. Type: String
          - pointsize (standard value: 20): Size of the spheres that represent
          the point clouds. Type <int>
          - colorpoints (standard value: "#c8102e"): Color of the spheres.
          - cam_az (standard value: 0): Camera azimuthal position.
          Type: <float (1)>
          - cam_el (standard value: 0): Camera elevation. Type: <float (1)>
          - wsize (standard value (3440, 1440)): Size of the window in pixels
          to plot the image. Type: <tuple (int, int)>

        Output:
          - None. The image is directly saved in the specified path.
        '''
        pltv = pv.Plotter(off_screen=True, window_size=wsize)
        pltv.set_background("white")
        pltv.add_mesh(Splot, point_size=pointsize, style="points",
                      render_points_as_spheres=True,
                      lighting=True, color=colorpoints)
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        pltv.view_isometric()
        pltv.camera.azimuth= cam_az
        pltv.camera.elevation= cam_el
        pltv.show(screenshot=figname)
        pltv.close()
        return()
    
    # Plot color maps onto 3D point clouds
    def pccmap(Splot, cscalar, figname, pointsize=20, vlim=[-1,1],
               cam_az=0, cam_el=0, wsize=(3440, 1440)):
        ''' Function for visualizing 3D point clouds with color maps.

        Input:
          - Splot: Array with the Cartesian coordinates of the 3D point cloud.
          Type: <array, (N,3)>
          - cscalar: Array with the scalar values associated with each point.
          Type: <array, (N)>
          - figname: Name of the file to save the figure. Type: String
          - pointsize (standard value: 20): Size of the spheres that represent
          the point clouds. Type <int>
          - vlim (standard value: [-1,1]): Minimum and maximum values of the
          color scale. Type: <list, [min, max]>
          - colorpoints (standard value: "#c8102e"): Color of the spheres.
          - cam_az (standard value: 0): Camera azimuthal position.
          Type: <float (1)>
          - cam_el (standard value: 0): Camera elevation. Type: <float (1)>
          - wsize (standard value (3440, 1440)): Size of the window in pixels
          to plot the image. Type: <tuple (int, int)>

        Output:
          - None. The image is directly saved in the specified path. 
        '''

        pltv = pv.Plotter(off_screen=True, window_size=wsize)
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
        pltv.view_yz()
        pltv.camera.azimuth= cam_az
        pltv.camera.elevation= cam_el
        pltv.show(screenshot=figname)
        pltv.close()
        return()

    # Comparison between point clouds
    def pccomp(Sref, Scomp, figname, pointsize=20, cscalar=None, vlim=[0,1],
               cam_az=0, cam_el=0, wsize=(3440, 1440)):
        ''' Function for comparing two point clouds.

        Input:
          - Sref: Array with the Cartesian coordinates of the reference 
          3D point cloud.
          Type: <array, (N,3)>
          - Scomp: Array with the Cartesian coordinates of the target 3D point cloud.
          Type: <array, (N,3)>
          - cscalar: Array with the scalar values associated with each point.
          Type: <array, (N)>
          - figname: Name of the file to save the figure. Type: String
          - pointsize (standard value: 20): Size of the spheres that represent
          the point clouds. Type <int>
          - vlim (standard value: [-1,1]): Minimum and maximum values of the
          color scale. Type: <list, [min, max]>
          - colorpoints (standard value: "#c8102e"): Color of the spheres.
          - cam_az (standard value: 0): Camera azimuthal position.
          Type: <float (1)>
          - cam_el (standard value: 0): Camera elevation. Type: <float (1)>
          - wsize (standard value (3440, 1440)): Size of the window in pixels
          to plot the image. Type: <tuple (int, int)>

        Output:
          - None. The image is directly saved in the specified path. 
        '''

        pltv = pv.Plotter(off_screen=True, 
                          window_size=wsize)
        pv.set_plot_theme("document")
        pltv.set_background("white")
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        
        pltv.add_mesh(Sref, point_size=pointsize, style="points",
                      render_points_as_spheres=True,
                      lighting=True, color="#00334c")
        Sref_yspan = np.max(Sref[:,1]) - np.min(Sref[:,1])
        if not type(cscalar)==type(None):
            pltv.add_mesh(Scomp+[0,2*Sref_yspan,0], 
                          point_size=pointsize, style="points", 
                          render_points_as_spheres=True,
                          lighting=True, scalars=cscalar.flatten(),
                          cmap=cm_custm,
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
        
        pltv.view_yz()
        pltv.camera.azimuth= cam_az
        pltv.camera.elevation= cam_el
        pltv.show(screenshot=figname)
        pltv.close()
        return()

    # Visualize 3D meshes
    def meshplot(mesh_path, figname, colormesh="#959da8", 
               cam_az=0, cam_el=0, wsize=(3440, 1440)):
        ''' Function for comparing two point clouds.

        Input:
          - mesh_path: path to the mesh to be visualized. Type <string>
          - figname: Name of the file to save the figure. Type: String
          - colormesh (standard value: "#959da8"): Color of the mesh.
          - cam_az (standard value: 0): Camera azimuthal position.
          Type: <float (1)>
          - cam_el (standard value: 0): Camera elevation. Type: <float (1)>
          - wsize (standard value (3440, 1440)): Size of the window in pixels
          to plot the image. Type: <tuple (int, int)>

        Output:
          - None. The image is directly saved in the specified path. 
        '''

        pltv = pv.Plotter(off_screen=True, window_size=wsize)
        pltv.set_background("white")
        pltv.add_mesh(pv.PolyData(mesh_path), lighting=True, color=colormesh)
        light = pv.Light(position=(1, 0, 0), light_type='scene light')
        pltv.add_light(light)
        pltv.view_isometric()
        pltv.camera.azimuth= cam_az
        pltv.camera.elevation= cam_el
        pltv.show(screenshot=figname)
        pltv.close()
        return()

# ------------------------------------------------------------------------------
# DESIGN APPLICATIONS
# ------------------------------------------------------------------------------

# Free-form deformation algorithm
# Standard free-form deformation (FFD) algorithm
class FFD:
    # FFD lattice
    def FFD_lattice(S,limits,L,M,N):
        '''Function to generate the representation of the control lattice and
        normalized point cloud representation

        Input:
          - S: 3D point cloud representation. Type <array, (N,3)>
          - limits: array with the bounding values of the x, y and z 
          coordinates -> [[x_min,x_max],[y_min,y_max],[z_min,z_max]].
          Type: <array, (3,2)>
          - L: Number of control planes in the s direction. Type: <int>
          - M: Number of control planes in the t direction. Type: <int>
          - N: Number of control planes in the u direction. Type: <int>

        Output:
          - Sn: 3D point cloud normalized to [0,1], [0,1], [0,1]. Type
          <array, (N,3)>
          - V: Coordinates of the control points. Type <array, (Nv,3)>

        '''
        # Axes seed
        x_seed = np.linspace(np.min(limits[0,:]), np.max(limits[0,:]), L)
        y_seed = np.linspace(np.min(limits[1,:]), np.max(limits[1,:]), M)
        z_seed = np.linspace(np.min(limits[2,:]), np.max(limits[2,:]), N)

        # Generate lattice
        V = np.zeros((L*M*N, 3))
        cntrV = 0
        for i in range(L):
            for j in range(M):
                for k in range(N):
                    V[cntrV, :] = np.array([x_seed[i], y_seed[j], z_seed[k]])
                    cntrV+=1

        # Embedd point cloud
        Sn = np.zeros(S.shape)
        for i in range(3):
            Sn[:,i] = (S[:,i]-np.min(V[:,i]))/(np.max(V[:,i])-np.min(V[:,i]))
        
        return(Sn, V)
    
    # Free-form deformation: Bernstein coefficients
    def BernsteinPoly(L,M,N,Sn):
        '''Function to generate the matrix with the coefficients
        of the Bernstein tri-variate polynomial

        Input:
          - L: Number of control planes in the s direction. Type: <int>
          - M: Number of control planes in the t direction. Type: <int>
          - N: Number of control planes in the u direction. Type: <int>
          - Sn: Normalized coordinates of the 3D point cloud 
          representation. Type: <array, (N,3)>

        Output:
          - B_marix: matrix with the coefficients of the tri-variate Bernstein
          polynomial. Type <array, (N, Nv)>
        '''
        # Normalized point cloud
        s = Sn[:,0]
        t = Sn[:,1]
        u = Sn[:,2]
    
        # Bernstein matrix
        shpBern = np.max([L, M, N])
        Mbernstein = np.matrix(np.zeros([shpBern, shpBern]))
        Mbernstein[0,0] = 1
        for i in range(1, Mbernstein.shape[0]):
            for j in range(Mbernstein.shape[1]):
                if j == 1: Mbernstein[i,j] = 1
                Mbernstein[i,j] = Mbernstein[i-1, j-1] + Mbernstein[i-1, j]
    
        # Polynomial coefficients
        B_matrix = np.zeros((s.shape[0], L*M*N))
        cnt = 0
        for i in range (L):
            s_term = Mbernstein[L-1,i]*(1-s)**(L-1-i)*s**i
            for j in range(M):
                t_term = Mbernstein[M-1,j]*(1-t)**(M-1-j)*t**j
                for k in range(N):
                    B_matrix[:, cnt] = s_term*t_term*(Mbernstein[N-1,k]*(1-u)**(N-1-k)*u**k)
                    cnt += 1
        return(B_matrix)

    # FFD operator
    def FFD_op(V, B):
        ''' Deformation of a 3D shape using FFD
        
        Input:
          - V: Matrix of control points. Type <array, (Nv,3)>
          - B: Coefficients of the tri-variate Bernstein polynomial.
          Type <array, (N, Nv)>

        Output:
          - S_out: Deformed 3D point cloud. Type <array, (N,3)>
        '''
        S_out = np.matmul(B,V)
        return(S_out)

# Algorithms for generating latent representations, 3D point clouds and 
# visualizing network features
class DesignApps:
    # Function to import the network graph
    def import_net_graph(config_path, GPUid):
        ''' Function to load graph nodes and start a tensorflow session
        that can be used for GDL applications with the trained architectures.

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - sess: Tensorflow session that should be used
          - S_in: Tensorflow placeholder for the input point clouds
          - Z: Tensor of the latent representation. Type: <tensor, (-1, Lz, 1)>
          - S_out: Tensor with the output point clouds. 
          Type: <tensor, (-1,pc_size,3)>
          - feat_layer: Tensor with the set of feature values (activations) 
          of the processed point clouds obtained at the last convolutional 
          layer. Type: <tensor, (-1, pc_size, n_features)>
          - pc_size: Size of the output point cloud. Type int
          - dpout: Tensorflow placeholder for the dropout ratio utilized in the PC-VAE. Type <tensor ()>
          - gamma_n: Tensorflow placeholder for enabling or disabling the 
          Gaussian noise in the latent layer. Type <tensor ()>.
          - latt_def: Tensor of deformed control points that were utilized to
          generate the output shape. Type <tensor (-1,n_control_points,3)>
          - flags: List with arrays that indicate the architecture that is 
          utilized. Type <list, [Boolean (VAE), Boolean (Point2FFD)]>
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

        ## Load the architecture
        flag_vae = False
        flag_p2ffd = False
        dpout = None
        gamma_n = None
        latt_def = None
        # In case the network is PC-AE
        try:
            # - Import Graph at latest state (after training)
            TFmetaFile = str.format("{}/pcae.meta", net_dir)
            TFDirectory = str.format("{}/", net_dir)
            # import graph data
            new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                   clear_devices=True)
        except:
            try:
                # In case the network is PC-VAE
                # - Import Graph at latest state (after training)
                TFmetaFile = str.format("{}/pcvae.meta", net_dir)
                TFDirectory = str.format("{}/", net_dir)
                # import graph data
                new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                       clear_devices=True)
                flag_vae = True
            except:
                # In case the network is Point2FFD
                # - Import Graph at latest state (after training)
                TFmetaFile = str.format("{}/p2ffd.meta", net_dir)
                TFDirectory = str.format("{}/", net_dir)
                # import graph data
                new_saver = tf.train.import_meta_graph(TFmetaFile,
                                                       clear_devices=True)
                dpout=None
                flag_p2ffd = True
        
        sess=tf.Session()
        new_saver.restore(sess, tf.train.latest_checkpoint(TFDirectory))
        graph = tf.get_default_graph()
        # Import network layers
        # Input
        S_in = graph.get_tensor_by_name("S_in:0")
        # - Latent representation
        Z = graph.get_tensor_by_name("Z:0")
        S_out = graph.get_tensor_by_name("S_out:0")
        pc_size = S_out.shape[1]
        # Droput (PC-VAE)
        if flag_vae: dpout = graph.get_tensor_by_name("do_rate:0")
        if flag_p2ffd:
            gamma_n = graph.get_tensor_by_name("gamma_n:0")
            latt_def = graph.get_tensor_by_name("Vd:0")

        layer_index = len(list(config["encoder_layers"]))
        feat_layer = graph.get_tensor_by_name(
                                 str.format("enclayer_{}:0", layer_index))

        flags = [flag_vae, flag_p2ffd]

        return(sess, S_in, Z, S_out, feat_layer, pc_size, dpout,
               gamma_n, latt_def, flags)

    # Calculate the latent representations of 3D point clouds
    def pointcloud_to_Z(config_path, sess, Sin, Z, pc_batch, flags,
                        dpout=None, gamma_n=None, GPUid=-1):
        '''Function to compress 3D point clouds to latent representations
        using a trained architecture.

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - sess: Tensorflow session that should be used
          - Sin: Tensorflow placeholder for the input point clouds
          - Z: Tensor of the latent representation. Type: <tensor, (-1, Lz, 1)>
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
          - flags: List with arrays that indicate the architecture that is 
          utilized. Type <list, [Boolean (VAE), Boolean (Point2FFD)]>
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

        # Architecture flags
        flag_vae, flag_p2ffd = flags
        
        ## Calculate reconstruction losses and latent representations
        print("\n")
        # Array of latent representations
        z_batch = np.zeros((data_set.shape[0], Z.shape[1], 1))
        for i in range(nshapes):
            print('Shape {} of {}'.format(i+1, nshapes), end="\r")
            xin = np.reshape(data_set[i,:,:], (1, -1, 3))
            # Calculate the latent representation and point cloud 
            # reconstruction
            if flag_vae:
                z_batch[i,:,:] = sess.run(Z,\
                    feed_dict={Sin: xin, dpout: 1.0})[0,:,:]
            else:
                if flag_p2ffd:
                    z_batch[i,:,:] = sess.run(Z, feed_dict={
                                                 Sin: xin, gamma_n:0})[0,:,:]
                else:
                    z_batch[i,:,:] = sess.run(Z, feed_dict={Sin: xin})[0,:,:]

        return(z_batch)
    
    # Generate 3D point clouds from latent representations
    def Z_to_pointcloud(config_path, sess, S_out, Z, z_batch, flags,
                        dpout=None, gamma_n=None, latt_def=None, GPUid=-1):
        '''Function to generate 3D point clouds from a batch of latent 
        representations.

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - sess: Tensorflow session that should be used
          - S_out: Tensor with the output point clouds. 
          Type: <tensor, (-1,pc_size,3)>
          - Z: Tensor of the latent representation. Type: <tensor, (-1, Lz, 1)>
          - z_batch: Array with the corresponding batch of latent 
          representations calculated for the input pc_batch.
          Type: <array (-1,Lz,1)>
          - flags: List with arrays that indicate the architecture that is 
          utilized. Type <list, [Boolean (VAE), Boolean (Point2FFD)]>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.

        Output:
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
          - V_batch: Batch of deformed control points that were utilized to
          generate the output shape. Type <array (-1,n_control_points,3)>
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
        # Point cloud size
        pc_size = S_out.shape[1]

        # Architecture flags
        flag_vae, flag_p2ffd = flags

        ## Calculate reconstruction losses and latent representations
        print("\n")
        # Batch of point clouds
        pc_batch = np.zeros((z_batch.shape[0], pc_size, 3))
        # Batch of control points (Point2FFD)
        if flag_p2ffd:
            V_batch = np.zeros((z_batch.shape[0], latt_def.shape[1], 3))
        else:
            V_batch = np.zeros((z_batch.shape[0], 1, 3))

        for i in range(nshapes):
            print('Shape {} of {}'.format(i+1, nshapes), end="\r")
            zin = np.reshape(z_batch[i,:,:], (1, -1, 1))
            if flag_vae:
                pc_batch[i,:,:] = sess.run(S_out,\
                    feed_dict={Z: zin, dpout: 1.0})[0,:,:]
            else:
                if flag_p2ffd:
                    pc, latt= sess.run([S_out, latt_def],\
                                        feed_dict={Z: zin, gamma_n: 0.0})
                    pc_batch[i,:,:] = pc[0,:,:]
                    V_batch[i,:,:] = latt[0,:,:]
                else:
                    pc_batch[i,:,:] = sess.run(S_out, feed_dict={
                                                         Z: zin})[0,:,:]

        # Load normalization limits
        normlim = np.load("{}/norm_inp_limits.npy".format(net_dir))
        pc_batch = CAE2PC.data_set_norm(pc_batch,
                                             inp_lim=np.array([0.1, 0.9]),
                                            out_lim=normlim)[0]
        V_batch = CAE2PC.data_set_norm(V_batch,
                                            inp_lim=np.array([0.1, 0.9]),
                                            out_lim=normlim)[0]
        # Output shapes
        return(pc_batch, V_batch)

    # Feature Visualization
    def featvis(config_path, sess, S_in, feat_layer, pc_batch, flags, 
                dpout=None, gamma_n=None, GPUid=-1, plot=True):
        ''' Function for visualizing the features learned by the PC-AE in the
        last convolutional layer (prior to max-pooling).

        Input:
          - config_path: Path to the dictionary (.py) with the settings for
          training the autoencoder. Type: <string>
          - sess: Tensorflow session that should be used
          - S_in: Tensorflow placeholder for the input point clouds
          - feat_layer: Tensor of the latent representation. 
          Type: <tensor, (-1, Lz, 1)>
          - pc_batch: Batch with 3D point clouds for calculating the latent 
          representations. Type: <array (-1,N,3)>
          - flags: List with arrays that indicate the architecture that is 
          utilized. Type <list, [Boolean (VAE), Boolean (Point2FFD)]>
          - GPUid (default=-1): ID of the GPU that will be used. If no GPU is
          avaliable, the value '-1' allows to train the model on CPU.
          - plot (default=True): Option that enables plot and store the
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

        # Architecture flags
        flag_vae, flag_p2ffd = flags

        ## Create directory to save the files
        output_test = str.format("{}/network_verification", net_dir)
        if not os.path.exists(output_test):
            os.system(str.format("mkdir {}", output_test))
            
        ## Calculate reconstruction losses and latent representations
        print("\n")
        # Array of latent representations
        f_batch = np.zeros((data_set.shape[0], feat_layer.shape[1], 
                                               feat_layer.shape[2]))
        for i in range(nshapes):
            print('Shape {} of {}'.format(i+1, nshapes), end="\r")
            xin = np.reshape(data_set[i,:,:], (1, -1, 3))
            # Calculate the latent representation and point cloud 
            # reconstruction
            if flag_vae:
                f_batch[i,:,:] = sess.run(feat_layer,\
                    feed_dict={S_in: xin, dpout: 1.0})[0,:,:]
            else:
                if flag_p2ffd:
                    f_batch[i,:,:] = sess.run(feat_layer, 
                                              feed_dict={S_in: xin, 
                                                  gamma_n:0})[0,:,:]
                else:
                    f_batch[i,:,:] = sess.run(feat_layer, 
                                              feed_dict={S_in: xin})[0,:,:]
        return(f_batch)