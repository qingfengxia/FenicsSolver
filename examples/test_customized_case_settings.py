# -*- coding: utf-8 -*-
# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2017 - Qingfeng Xia <qingfeng.xia iesensor.com>         *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU Lesser General Public License (LGPL)    *
# *   as published by the Free Software Foundation; either version 2 of     *
# *   the License, or (at your option) any later version.                   *
# *   for detail see the LICENCE text file.                                 *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU Library General Public License for more details.                  *
# *                                                                         *
# *   You should have received a copy of the GNU Library General Public     *
# *   License along with this program; if not, write to the Free Software   *
# *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  *
# *   USA                                                                   *
# *                                                                         *
# ***************************************************************************

from __future__ import print_function, division

"""
This is outdated, served solver for FreeCAD CFD GUI 
"""

from config import is_interactive
interactively = is_interactive()  # manual set it False to debug solver

from FenicsSolver.main import load_settings
from FenicsSolver import ScalarTransportSolver

def test_elasticity():
    pass

def test_CFD():
    #
    raise NotImplementedError("currently, CFD test is not usable, try CFD test in examples folder")
    df = '../data/TestCFD.json'
    settings = load_settings(df)  # force python2 load ascii string
    #settings['case_folder'] = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + "data"
    #
    from FenicsSolver import CoupledNavierStokesSolver
    solver = CoupledNavierStokesSolver.CoupledNavierStokesSolver(settings)
    solver.print()
    solver.solve()
    solver.plot()

def test_heat_transfer():
    #print(os.path.dirname(__file__))  # __file__ is absolute path for python 3.4+
    df = '../data/TestHeatTransfer.json'
    settings = load_settings(df)  # load FreeCAD GUI generated json data
    #settings['case_folder'] = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + "data"
    ###########################################
    """ here lot of customization can be done, e.g. 
    settings['body_source'] = dolfin.Expression('', degree=1)
    anisotropic material, convective velociyt, see examples/test_*.py for more details
    """
    ###########################################
    solver = ScalarTransportSolver.ScalarTransportSolver(settings)
    solver.print()
    solver.solve()
    solver.plot()

if __name__ == "__main__":
    test_heat_transfer()