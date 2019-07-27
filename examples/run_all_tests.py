"""local test before package generation and push to pypi
"""

from __future__ import print_function, division

import os.path, os, sys

import glob
test_files = glob.glob("test_*.py")

this_dir = os.path.dirname(os.path.realpath(__file__))
#currently, not all test can pass the test, solvers are under development
test_files = """test_cfd_solver.py, test_heat_transfer.py, test_large_deformation.py, test_nonlinear_elasticity.py, test_electrostatics.py"""
#test_plasticity.py, test_flow_pass_cylinder.py, test_customized_case_settings.py
#test_linear_elasticity.py, takes long time to complete
test_files = test_files.split(', ')

os.environ['FENICSSOLVER_BATCH'] = 'TRUE'

for tf in test_files:
    print("run the test file:", tf)
    exec(open(this_dir + os.path.sep + tf).read())  # compatible for both py2 and py3

# unset env var,  os.unsetenv() does not work on OSX
del os.environ['FENICSSOLVER_BATCH']