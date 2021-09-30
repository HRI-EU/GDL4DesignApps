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

# SCRIPT FOR TESTING THE FUNCTIONALITY OF THE SCRIPTS

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

# Authors: Thiago Rios, Sneha Saha

## FAILURE CHECKS
# TEST 1: Check if the library is installed
dtest=$(pwd)
cd ..
echo -e "# TEST 1: Check if library was correctly installed" >> $dtest/log_utest.dat
echo -e "Message in case of error: ModuleNotFoundError: No module names gdl4designapps" >> $dtest/log_utest.dat
echo -e "Run test ..." >> $dtest/log_utest.dat
python -c "import gdl4designapps" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo -e "Finished test 1" >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat

# TEST 2: Load individual components of the library
echo -e "# TEST 2: Load individual components of the library" >> $dtest/log_utest.dat
echo -e "Message in case of error: ModuleNotFoundError: No module names gdl4designapps" >> $dtest/log_utest.dat
echo -e "Load preprocess_methods..." >> $dtest/log_utest.dat
python -c "from gdl4designapps import preprocess_methods" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo -e "Load preprocess.CAE2PC..."  >> $dtest/log_utest.dat
python -c "from gdl4designapps.preprocess_methods import CAE2PC" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat

echo -e "Load designapps..." >> $dtest/log_utest.dat
python -c "from gdl4designapps import designapps" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo -e "Load designapps.Vis3D..." >> $dtest/log_utest.dat
python -c "from gdl4designapps.designapps import Vis3D" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo -e "Finished Test 2" >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat

# TEST 3: Run basic scripts
echo -e "# TEST 3: Running basic scripts" >> $dtest/log_utest.dat
echo -e "Load data set" >> $dtest/log_utest.dat
echo -e "Expected error: Path to data set not found" >> $dtest/log_utest.dat
python -c "from gdl4designapps.preprocess_methods import CAE2PC; out = CAE2PC.pc_sampling('testdata', 2048)" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat 
echo " " >> $dtest/log_utest.dat

echo -e "Generate PC-AE graph" >> $dtest/log_utest.dat
echo -e "Expected output: Graph variables" >> $dtest/log_utest.dat
python -c "from gdl4designapps.preprocess_methods import PC_AE; out = PC_AE.pcae([64, 128, 128, 256], 2048, 128, [256, 256]); print(out)" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat

echo -e "Start PC-AE training" >> $dtest/log_utest.dat
echo -e "Expected error: configuration file not found!" >> $dtest/log_utest.dat
python -c "from gdl4designapps.preprocess_methods import arch_training; out = arch_training.pc_ae_training('testconfig')" >> $dtest/log_utest.dat
echo -e "Done!" >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat
echo " " >> $dtest/log_utest.dat

cd $dtest

## TEST CONCLUDED
echo -e "unittest concuded!" >> log_utest.dat
echo -e "If the tests returned no unexpected messages, the library was sucessfully installed and is ready for use." >> log_utest.dat

echo -e "###### TEST CONCLUDED ######" >> log_utest.dat
echo "###### TEST CONCLUDED ######"
# EOF
