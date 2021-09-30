#!/usr/bin/env python
'''
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

Script for installing the GDL4DesignApps library

Copyright (c)
Honda Research Institute Europe GmbH

Authors: Thiago Rios, Sneha Saha
Contact: gdl4designapps@honda-ri.de
'''

from distutils.core import setup

setup(name='gdl4designapps',
      version='1.0.0',
      description='Geometric deep learning models for engineering design applications.',
      author='Thiago Rios, Sneha Saha',
      author_email='gdl4designapps@honda-ri.de',
      url='https://github.com/HRI-EU/GDL4DesignApps',
      packages=['gdl4designapps'],
      license='GPLv3.0'
     )
#EOF