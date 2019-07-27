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
import math
import numpy as np

from dolfin import *

"""
https://github.com/Jiangliuer/Fenics/tree/master/DG
"""

magnetic_permeability_0 = 4 * pi * 1e-7
electric_permittivity_0 = 8.854187817e-12

from SolverBase import SolverBase, SolverError
class MaxwellEMSolver(SolverBase):
    # 
    def __init__(self, s):
        SolverBase.__init__(self, s)
        self.settings['vector_name'] = 'magnetic_potential_vector'

    def set_function_space(self, mesh_or_function_space, periodic_boundary):
        try:
            self.mesh = mesh_or_function_space
            if periodic_boundary:
                self.function_space = VectorFunctionSpace(self.mesh, "CG", self.degree, constrained_domain=periodic_boundary)
                # the group and degree of the FE element.
            else:
                self.function_space = VectorFunctionSpace(self.mesh, "CG", self.degree)
        except:
            self.function_space = mesh_or_function_space
            self.mesh = self.function_space.mesh()
        self.is_mixed_function_space = False  # how to detect it is mixed, vector, scaler , tensor?

    def permeability(self):
        return magnetic_permeability_0
    
    def permittivity(self):
        return electric_permittivity_0

    def update_boundary_conditions(self, time_iter_, u, v, ds):
        V = self.function_space
        bcs = []
        integrals_F = []
        mesh_normal = FacetNormal(self.mesh)  # n is predefined as normal?
        for name, bc_settings in self.boundary_conditions.items():
            i = bc_settings['boundary_id']
            bc = self.get_boundary_variable(bc_settings)
            
            if bc['type'] =='Dirichlet': # current density
                
            elif bc['type'] =='Neumann':
                # tangential B (magnetic_field_vector)
            else:
            
        return bcs, integrals_F

    def generate_form(self, time_iter_, u, v, u_current, u_prev):
        V = self.function_space  # the vector u is electric field
        omega = Constant('1.0')
        #f is source, 
        F = inner(curl(u),curl(v))*dx - omega**2*inner(u,v)*dx - inner(f,v)*dx
        return F, bcs

    def solve_form(self, F, u_, bcs):
        #if self.is_iterative_solver:
        #u_ = self.solve_iteratively(F, bcs, u)
        u_ = self.solve_amg(F, u_, bcs)
        # calc boundingbox to make sure no large deformation?
        return u_

    def solve(self):
        u = self.solve_transient()  # defined in SolverBase
        return u