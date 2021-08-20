# ## LICENSE: GPL 3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or 
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# SCRIPT FOR TESTING THE FUNCTIONALITY OF THE AUTOENCODER ALGORITHMS

# Pre-requisites:
#  - Python      3.6.10
#  - numpy       1.19.1
#  - TensorFlow  1.14.0
#  - TFLearn     0.3.2
#  - cudatoolkit 10.1.168
#  - cuDNN       7.6.5
#  - Ubuntu      18.04
#  - pandas      1.1.0

# Copyright (c)
# Honda Research Institute Europe GmbH


# Authors: Thiago Rios <thiago.rios@honda-ri.de>

## FAILURE CHECKS
# FAIL 1: non-existing data set, autoencoder training
echo "TEST 1a: autoencoder training, non-existing data set" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Script interrupted. Directory not found!" >> log_uteest.dat
echo "--- Path to data set: NonExst" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/pcae_training.py --N 2048 --LR 128 --GPU -1 --i NonExst >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo  >> log_uteest.dat

echo "TEST 1b: variational autoencoder training, non-existing data set" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Script interrupted. Directory not found!" >> log_uteest.dat
echo "--- Path to data set: NonExst" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/vpcae_training.py --N 2048 --LR 128 --GPU -1 --i NonExst  >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo   >> log_uteest.dat
echo   >> log_uteest.dat
echo   >> log_uteest.dat

# FAIL 2: non-existing training directory, autoencoder testing
echo "TEST 2a: reconstruction loss algorithm, training directory missing" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Directory Network_pcae_N0_LR0 does not exist!" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/evaluate_loss.py --N 0 --LR 0 --GPU -1 --i NonExst >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo  >> log_uteest.dat

echo "TEST 2b: point cloud generation, training directory missing" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Directory Network_pcae_N0_LR0 does not exist!" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/pointcloud_generator.py --N 0 --LR 0 --GPU -1 --i NonExst >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo  >> log_uteest.dat

echo "TEST 2c: variational autoencoder testing, training directory missing" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Directory Network_v_pcae_N0_LR0 does not exist!" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/evaluate_loss.py --N 0 --LR 0 --GPU -1 --i NonExst --VAE True >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo  >> log_uteest.dat

echo "TEST 2d: point cloud generation, training directory missing" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Directory Network_v_pcae_N0_LR0 does not exist!" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/pointcloud_generator.py --N 0 --LR 0 --GPU -1 --i NonExst --VAE True >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo   >> log_uteest.dat
echo   >> log_uteest.dat
echo   >> log_uteest.dat

# FAIL 3: non-existing training directory, extraction of weights
echo "TEST 3a: autoencoder, weights extraction, training directory missing" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Directory Network_pcae_N0_LR0 does not exist!" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/ae_parameters_extraction.py --N 0 --LR 0 --GPU -1 >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo  >> log_uteest.dat

echo "TEST 3b: variational autoencoder, weights extraction, training directory missing" >> log_uteest.dat
echo "Standard output message:" >> log_uteest.dat
echo "--- Directory Network_v_pcae_N0_LR0 does not exist!" >> log_uteest.dat
echo   >> log_uteest.dat
python3 include/ae_parameters_extraction.py --N 0 --LR 0 --GPU -1 --VAE True >> log_uteest.dat
echo --------------------------------------------------------- >> log_uteest.dat
echo   >> log_uteest.dat
echo   >> log_uteest.dat
echo   >> log_uteest.dat

## TEST CONCLUDED
echo "###### TEST CONCLUDED ######"
# EOF
