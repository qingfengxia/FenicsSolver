from __future__ import print_function, division
import os
import sys
import argparse
import configuration_file as config

import PySolverInterface as PSI
from PySolverInterface import *

parser = argparse.ArgumentParser()
parser.add_argument("configurationFileName", help="Name of the xml precice configuration file.", type=str)

"""
generate the preCICE xml config by this class
example by coupling foam solver
coupling dict, mapping to preCICE config
"""
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("configurationFileName", help="Name of the xml precice configuration file.", type=str)

    try:
        args = parser.parse_args()
    except SystemExit:
        print("")
        print("Did you forget adding the precice configuration file as an argument?")
        print("Try $python FluidSolver.py precice-config.xml")
    quit()

def init_precice():
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       print "ERROR: PRECICE_ROOT not defined!"
       exit(1)

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)