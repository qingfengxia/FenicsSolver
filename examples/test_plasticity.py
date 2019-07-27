# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2018 - Qingfeng Xia <qingfeng.xia eng ox ac uk>                 *       *
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

"""
This solver is not completed:
  File "/media/sf_OneDrive/gitrepo/FenicsSolver/examples/../FenicsSolver/SolverBase.py", line 511, in solve_transient
    self.solve_current_step()
TypeError: solve_current_step() takes exactly 5 arguments (1 given)

"""

from config import is_interactive
interactively = is_interactive()

from dolfin import *
set_log_level(ERROR)
from FenicsSolver import PlasticitySolver
from FenicsSolver import SolverBase


def test():
    #
    mesh=Mesh("beam10000.xml.gz")
    print(mesh)
    # overlapping boundary in the original example, adapted
    DirichletXY = CompiledSubDomain("near(x[0],0.) && near(x[2],0.5) && on_boundary")
    DirichletZ = CompiledSubDomain("(near(x[0], 5.0) && near(x[2], 0.5)) && on_boundary")

    from collections import OrderedDict
    bcs = OrderedDict()
    #
    bcs["left"] = {'boundary': DirichletXY, 'boundary_id': 1, 'type': 'Dirichlet', 'value': (Constant(0.), Constant(0.), Constant(0.))}
    bcs["right"] = {'boundary': DirichletZ, 'boundary_id': 2, 'type': 'Dirichlet', 'value': (None, None, Constant(0.))}

    import copy
    s = copy.copy(SolverBase.default_case_settings)  # deep copy? 
    s['material'] = {'name': 'rubber', 'elastic_modulus': 10, 'poisson_ratio': 0.3, 
                                'density': 800, 'thermal_expansion_coefficient': 2e-6}  # fixme, unknown properties value
    s['mesh'] = mesh
    s['boundary_conditions'] = bcs
    s['body_source'] = Constant((0, 0, -1))  # lambda t : (0, 0, -40.0*sin(t/2.*pi))
    s['solver_settings'] = {
        'transient_settings': {'transient': True, 'starting_time': 0, 'time_step': 0.05, 'ending_time': 1},
        'reference_values': {},
        }

    s['plasticity_model'] = {'name': 'hardening'}
    solver = PlasticitySolver.PlasticitySolver(s) # body force test passed

    u = solver.solve()
    ## Plot solution of displacement
    plot(u, title='displacement')
    # Plot stress, fixme:  von_Mises may not apply
    #plot(solver.von_Mises(u), title='Stress von Mises')

    #if not run by pytest
    if interactively:
        solver.plot()

if __name__ == '__main__':
    test()