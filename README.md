# Geometric Deep Learning for Design Applications
The present repository contains the software for training and utilizing the point cloud autoencoders in design applications. The architectures were developed in the framework of the ECOLE project (Horizon 2020 MSCA-ITN, Grant number 766186).

---

## Pre-requisites
The scripts of the repository were tested on Ubuntu 18.04 in a _conda_
environment with Python 3.6.10 and the following standard libraries installed:

+ numpy          1.19.1
+ pandas         1.1.0
+ tensorflow-gpu 1.14.0
+ TFLearn        0.3.2
+ matplotlib     3.2.2
+ plotly         4.9.0
+ cudatoolkit    10.1.168
+ cudnn          7.6.5
+ scikit-learn   0.23.2
+ plotly         4.9.0
+ pyvista        0.29.1

For training the autoencoder, we adapted the scripts of the loss functions
implemented in ([Achlioptas et al. 2018](https://github.com/optas/latent_3d_points))
to our Python version and installed the library in the same conda environment.

---

## How to cite
**3D Point Cloud Autoencoder**

Read our paper [here](https://ieeexplore.ieee.org/document/9446541)
```
@ARTICLE{Rios2021a,
  author={Rios, Thiago and van Stein, Bas and Bäck, Thomas and Sendhoff, Bernhard and Menzel, Stefan},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={{Multi-Task Shape Optimization Using a 3D Point Cloud Autoencoder as Unified Representation}}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TEVC.2021.3086308}}
```

**3D Point Cloud Variational Autoencoder**

Read our paper [here](https://www.honda-ri.de/publications/publications/?pubid=4510)
```
@INPROCEEDINGS{Saha2020,
  author    = {Saha, Sneha and Menzel, Stefan and Minku, Leandro L. and Yao, Xin and Sendhoff, Bernhard and Wollstadt, Patricia},
  booktitle = {2020 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
  title     = {{Quantifying The Generative Capabilities Of Variational Autoencoders For 3D Car Point Clouds}}, 
  year      = {2020},
  pages     = {1469-1477},
  doi       = {10.1109/SSCI47803.2020.9308513}
}
```

**Feature Visualization for 3D Point Cloud Autoencoders**

Read our paper [here](https://www.honda-ri.de/publications/publications/?pubid=4354)
```
@INPROCEEDINGS{Rios2020a,
  author      = {Rios, Thiago and van Stein, Bas and Menzel, Stefan and Back, Thomas and Sendhoff, Bernhard and Wollstadt, Patricia},
  booktitle   = {2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title       = {{Feature Visualization for 3D Point Cloud Autoencoders}}, 
  year        = {2020},
  pages       = {1-9},
  doi         = {10.1109/IJCNN48605.2020.9207326}
  }
```

---

## Software Modules
The scripts contained in this repository are the following:

### Auxiliary scripts
1. _autoencoder_architectures.py_ : It contains the necessary algorithms for
assigning the parameters of the autoencoders and generating the Tensorflow
graphs for running the models on CPU or GPU. This script is called from 
within the training/testing scripts.

2. _preproc_scripts.py_ : It contains auxiliary algorithms for preprocessing 
the point cloud data before starting the training or testing the autoencoders,
such as for loading the point cloud data. This script is also called from 
within the training/testing scripts.

### Main (training) scripts
3. _pcae_training.py_ : script for training the **vanilla** point cloud 
autoencoder. It is one of the main scripts and it requires the two previous 
files to be in the same directory and the implementation in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points) to be 
installed in the same conda environment.

4. _vpcae_training.py_ : script for training the **variational** point cloud 
autoencoder. It is one of the main scripts and it requires the two previous 
files to be in the same directory and the implementation in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points) to be 
installed in the same conda environment.

### Verification and post processing
5. _evaluate_loss.py_ : Algorithm that calculates the Chamfer Distance between
a set of input point clouds, given a list (text file), and the corresponding
reconstructions yielded by a trained (variational) point cloud autoencoder. The
output is stored in a text file in the training directory of the autoencoder, 
which contains the location of the geometries, latent representations and 
calculated Chamfer Distance. Running the script requires the implementation in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points) to be 
installed in the same conda environment.

6. _pointcloud_generator.py_ : Algorithm to reconstruct 3D point clouds based
on a set of latent representations, provided as a text file, and a trained
(variation) point cloud autoencoder. Running the script requires the 
implementation in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points) to be 
installed in the same conda environment.

7. _autoencoder_lr_interpolation_ : Script for interpolating shapes using the
trained autoencoders, taking five geometries from each training and test set as 
input. _This script is an example of potential application of the autoencoder._
Running the script requires the implementation in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points) to be 
installed in the same conda environment.

**Obs.:** The scripts were developed considering that the user can recover the
trained architectures from the `.META` files stored in the training directory
of the autoencoder. However, since the architecture depends on third-party
implementation, the scripts might fail if the installation is not performed
correctly. Therefore, we added a script for extracting the network parameters
and saving in text files, which can be read and assigned to clean architectures
for testing the autoencoders, without depending on the algorithm for the Chamfer
Distance.

8. _ae_parameters_extraction_: The scripts loads the architecture of a trained
point cloud autoencoder, reads the parameters and save the matrices as text
files in the autoencoder training directory.
Running the script requires the implementation in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points) to be 
installed in the same conda environment, if the Chamfer distance was used to
train the target autoencoder.

---

## How to use the scripts
The application of the scripts is divided in three steps: **pre-processing**,
**training** and **application**.

### Pre-processing and installation
The scripts that have 3D point clouds as input require a **data set folder**,
where the **geometries are stored as:**
+ **.csv** files, with the values of the coordinates delimited by commas and each
line defines a point;
+ **.xyz** files, where the coordinates are delimited by a single space and each
line defines a point;
+ **.stl** files, containing a single object each (i.e. not concatenated stls) in
the ASCII format;
+ **.obj** files, either as scene of object.

The preprocessing scripts were developed to handle directories comprising mixed
file formats, however it is not recommended. **The directory must not contain any**
**other file than the geometries used for training/testing**.

In case [ShapeNetCore](https://www.shapenet.org/) data is used, the models 
should be preprocessed and 
organized in a single directory. The current structure of the ShapeNetCore data
set is not supported by the current version of the algorithms in this repository.

For running the scripts, we recommend to set up an anaconda environment, which
can be performed in the terminal window with the following command:

```
conda create -n myenv tensorflow-gpu cudatoolkit=10.1.168 matplotlib libtiff libgcc libgcc-ng cudnn
```

Then, activate the same environment with the command `conda activate myenv` and
install the implementation provided in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points). 
As aforementioned, this implementation requires an adaptation of the scripts to 
Python 3.6 in order to run the algorithms implemented in this repository.

### Training the Autoencoders
One can train the *Vanilla* point cloud autoencoder on a point cloud data set
using the following command on a terminal window:

```
python pcae_training.py --N [point_cloud_size] --LR [latent_representation_size] --GPU [gpu_id] --i [data_set_directory] --o [output_directory]
```

In case the user wants to train the model using CPU instead of GPU, the 
parameter `[gpu_id]` should be set as `-1`. Assigning an `[output_directory]`
is not mandatory and, if no value is assigned, the script creates a directory
in the path that the user starts the training algorithm.

For the *variational* autoencoder, the process is similar, changing only the
name of the python script:

```
python vpcae_training.py --N [point_cloud_size] --LR [latent_representation_size] --GPU [gpu_id] --i [data_set_directory] --o [output_directory]
```

The outputs of the scripts are stored in a directory, following the structure:

```
Network_(v_)pcae_N{}_LR{}
|
+--checkpoint
+--geometries_testing.csv
+--geometries_training.csv
+--log_dictionary.py
+--normvalues.csv
+--(v)pcae.data-00000-of-00001
+--(v)pcae.index
+--(v)pcae.meta
+--(v_)pcae_N{}_LR{}_losses_test.csv
+--(v_)pcae_N{}_LR{}_losses_training.csv
+--plot_losses_(v_)pcae_N{}_LR{}.png
|
+--(v_pcae_N{}_LR{}_KL_losses_test.csv)
+--(v_pcae_N{}_LR{}_KL_losses_training.csv)
+--(v_pcae_N{}_LR{}_recon_losses_test.csv)
+--(v_pcae_N{}_LR{}_recon_losses_training.csv)
+--(training_set.pkl)
+--(test_set.pkl)
+--(validation_set.pkl)
```

Where the terms in parentheses are generated when the *variational* 
autoencoder is trained and the fields `{}` correspond to the values 
of the parameters assigned by the user.
Further network parameters are hard coded in the scripts and explained
with comments.

### Testing the Autoencoders
Once the architectures were trained, we can test them regarding the shape
reconstruction and generation capabilities. For calculating the Chamfer
Distance on a set of shapes, we can use the _evaluate_loss.py_ script with
the following command:

```
python evaluate_loss.py --N [point_cloud_size] --LR [latent_representation_size] --GPU [gpu_id] --i [list_with_pointclouds_path] --VAE [True/False]
```

where `[list_with_pointclouds_path]` specifies the path to a text file containing
the path to the point cloud files to be tested. **The point clouds must be**
**stored as .xyz files, otherwise the algorithm cannot load the point clouds**.
The parameter `--VAE` activates a flag that allows the algorithm to load the
architecture of a variational autoencoder, when the value `True` is assigned.
Otherwise, it is assumed `False` and the script loads the architecture of the
vanilla autoencoder trained with the same hyperparameters (_N, LR_).

The output of the script is a _.dat_ file containing the path of the tested
geometries, the corresponding latent variables and Chamfer Distance, following
the format:

```
path_to_geometry_0,lr_0,lr_1,...,lr_n,CD_0
path_to_geometry_1,lr_0,lr_1,...,lr_n,CD_1
...
path_to_geometry_j,lr_0,lr_1,...,lr_n,CD_j
```

A second approach for testing the architectures is generating point clouds from
a set of the latent variables. It is performed running the _pointcloud_generator.py_
script with the following command:

```
python pointcloud_generator.py --N [point_cloud_size] --LR [latent_representation_size] --GPU [gpu_id] --i [list_with_latent_variables] --VAE [True/False]
```

Where `[list_with_latent_variables]` is the path to a list of values for
the latent variables for generating the shapes. The values for each latent
variable are assigned in columns, delimited by commas, and each line of the
files specifies a shape. The remaining parameters have the same functions as
in the previous scripts.

The user can also extract the trained network parameters for further applications
running the following command:

```
python ae_parameters_extraction.py --N [point_cloud_size] --LR [latent_representation_size] --GPU [gpu_id] --VAE [True/False]
```

---

## Example: Interpolation in the latent space
For testing the trained autoencoders in an interpolation task, 
open the terminal and type
```
python autoencoder_lr_interpolation.py --N [pc_size] --LR [latent_space_dimension] --VAE [if variational, True] --GPU [gpu_to_be_assigned] --VAE [True/False]
```

**WARNING!** The script identifies the directory with the trained parameters
based on the template used in the training scripts 
([_pcae_training_, L#193](pcae_training.py#L193), and 
[_vpcae_training_, L#148](vpcae_training.py#L148)). If the pattern is modified
in the training files, it also has to be changed in the interpolation script.

---

## Trained Network Parameters
The directories _vanilla_pc_ae_parameters_ and _variational_pc_ae_parameters_
store the files with the trained parameters of the vanilla and variational
point cloud autoencoders, respectively, trained under the following conditions:

+ 3D point clouds with 2048 points, sampled from the car class of ShapeNetCore
with random uniform probability distribution.
+ 128-dimensional latent space.
+ Batch size, learning rate and further hyperparameters as described in [Rios et al. 2020](https://www.honda-ri.de/pubs/pdf/4354.pdf) and assigned in the _vpcae_training.py_.

The parameters stored in these directories allow further users test the architectures
as implemented for the papers, without depending on the library available in 
[Achlioptas et al. 2018](https://github.com/optas/latent_3d_points).

---

## Licensing
The software in this repository is licensed under the GPL 3.0. For more details on the license, please check [the license file.](LICENSE.md)