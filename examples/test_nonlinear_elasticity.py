# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2017 - Qingfeng Xia <qingfeng.xia eng ox ac uk>                 *       *
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
import math
import numpy as np

from dolfin import *
from FenicsSolver import LinearElasticitySolver
from FenicsSolver import NonlinearElasticitySolver
from FenicsSolver import SolverBase

interactively = True
set_log_level(ERROR)

def test():

    # Create mesh and define function space
    mesh = UnitCubeMesh(24, 16, 16)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    
    B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
    T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary, for all?

    # Mark boundary subdomians
    left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
    # Define Dirichlet boundary (x = 0 or x = 1)
    c = Constant((0.0, 0.0, 0.0))
    r = Expression(("scale*0.0",
                    "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                    "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                    scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3, degree=2)

    bcl = DirichletBC(V, c, left)
    bcr = DirichletBC(V, r, right)

    from collections import OrderedDict
    bcs = OrderedDict()
    #
    bcs["left"] = {'boundary': left, 'boundary_id': 1, 'type': 'Dirichlet', 'value': c}
    bcs["right"] = {'boundary': right, 'boundary_id': 2, 'type': 'Dirichlet', 'value': r}

    import copy
    s = copy.copy(SolverBase.default_case_settings)  # deep copy? 
    s['material'] = {'name': 'rubber', 'elastic_modulus': 10, 'poisson_ratio': 0.3, 
                                'density': 800, 'thermal_expansion_coefficient': 2e-6}  # fixme, unknown properties value
    s['function_space'] = V
    s['boundary_conditions'] = bcs
    s['body_source'] = B
    s['surface_source'] = {'value': Constant(0.1), 'direction': Constant((1, 0.0, 0.0)) }  #T  # apply to all boundaries, 
    solver = NonlinearElasticitySolver.NonlinearElasticitySolver(s)  # body force test passed
    #lsolver = LinearElasticitySolver.LinearElasticitySolver(s)
    #lu = lsolver.solve()

    u = solver.solve()
    ## Plot solution of displacement
    # Plot stress, fixme:  von_Mises may not apply
    #plot(solver.von_Mises(u), title='Stress von Mises')

    #if not run by pytest
    if interactively:
        solver.plot()

if __name__ == '__main__':
    test()