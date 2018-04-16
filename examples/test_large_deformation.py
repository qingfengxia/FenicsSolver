# -*- coding: utf-8 -*-
# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2018 - Qingfeng Xia <qingfeng.xia iesensor.com>         *
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

#from __future__ import print_function, division
import math
import numpy as np

from dolfin import *
from FenicsSolver.LargeDeformationSolver import LargeDeformationSolver
from FenicsSolver import SolverBase

def solve_elasticity(using_2d, length, E, nu, dt, t_end, dirname):
    """Prepares 2D geometry. Returns facet function with 1, 2 on parts of  the boundary."""
    if using_2d:
        n = 4
        x0 = 0.0
        x1 = x0 + length
        y0 = 0.0
        y1 = 1.0
        mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), int((x1-x0)*n), int((y1-y0)*n), 'crossed')
        gdim = 2
        bF_direction = Constant((0.0, 1.0))
    else:
        mesh = Mesh('lego_beam.xml')
        gdim = mesh.geometry().dim()
        x0 = mesh.coordinates()[:, 0].min()
        x1 = mesh.coordinates()[:, 0].max()
        gdim = 3
        bF_direction = Constant((0.0, 0.0, 1.0))

    left  = AutoSubDomain(lambda x: near(x[0], x0))
    right = AutoSubDomain(lambda x: near(x[0], x1))

    from collections import OrderedDict
    bcs = OrderedDict()
    # not supported value type: V.sub(0).sub(0)
    bcs["fixed"] = {'boundary': left, 'boundary_id': 1, 'type': 'Dirichlet', \
                            'value': (gdim*(0.0, ), gdim*(0.0, ))}
    bfunc = lambda t: 100*t # it should be a functon of time
    bcs["displ"] = {'boundary': right, 'boundary_id': 2, 'type': 'stress', 'value': bfunc, 'direction': bF_direction}

    import copy
    s = copy.copy(SolverBase.default_case_settings)
    s['material'] = {'name': 'steel', 'elastic_modulus': E, 'poisson_ratio': nu, 'density': 1000, 
                                'thermal_expansion_coefficient': 2e-6} #

    s['mesh'] = mesh
    s['boundary_conditions'] = bcs
    #s['temperature_distribution']=None
    #s['vector_name'] = ['displacement', 'velocity']
    s['solver_settings'] = {
        'transient_settings': {'transient': True, 'starting_time': 0, 'time_step': dt, 'ending_time': t_end},
        'reference_values': {'temperature': 293 },
        }

    # solver specific setting
    #
    solver = LargeDeformationSolver(s)
    w = solver.solve()
    solver.plot(w)

if __name__ == '__main__':
    solve_elasticity(True, 20, 1e5, 0.3, 0.25, 5, 'results_2d_comp')
    solve_elasticity(True, 20, 1e5, 0.5, 0.25, 5, 'results_2d_incomp')
    #solve_elasticity(geometry_2d(80.0), 1e5, 0.3, 0.25, 5.0, 'results_2d_long_comp')
    #solve_elasticity(False, 20, 1e5, 0.3, 0.50, 5.0, 'results_3d_comp')
    interactive()