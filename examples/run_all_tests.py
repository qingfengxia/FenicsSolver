"""local test before package generation and push to pypi
"""

from __future__ import print_function, division

import os.path, os, sys

import glob
test_files = glob.glob("test_*.py")

test_files = """test_cfd_solver.py, test_heat_transfer.py, test_large_deformation.py, test_nonlinear_elasticity.py, test_linear_elasticity.py, test_electrostatics.py"""
#test_plasticity.py, test_flow_pass_cylinder.py, test_customized_case_settings.py
test_files = test_files.split(', ')

for tf in test_files:
    print(tf)
    exec(open(tf).read())  # compatible for both py2 and py3