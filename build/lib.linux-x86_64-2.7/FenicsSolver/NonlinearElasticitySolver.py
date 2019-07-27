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
import collections
import numbers
import numpy as np


#####################################
from dolfin import *

from .SolverBase import SolverBase, SolverError
from .LinearElasticitySolver import LinearElasticitySolver

class NonlinearElasticitySolver(LinearElasticitySolver):
    """ share code as much as possible withe the linear version, especially boundary settings
    supported: nonlinear material property, hyperelastic material
    hyperelasticity: adapted from official tutorial:  
    https://github.com/FEniCS/dolfin/blob/master/demo/documented/hyperelasticity/python/demo_hyperelasticity.py.rst
    """
    def __init__(self, s):
        LinearElasticitySolver.__init__(self, s)
        #nonlinear special setting
        self.settings['mixed_variable'] = ('displacement', 'velocity', 'pressure')

    def generate_form(self, time_iter_, u, v, u_current, u_prev):
        # todo: transient
        # Optimization options for the form compiler
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["representation"] = "uflacs"
        V = self.function_space

        elasticity = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = elasticity/(2.0*(1.0 + nu))
        lmbda = elasticity*nu/((1.0 + nu)*(1.0 - 2.0*nu))

        I = Identity(self.dimension)             # Identity tensor
        F = I + grad(u_current)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor
        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

        # Total potential energy
        Pi = psi*dx 
        #- dot(T, u_current)*ds      # Traction force on the boundary
        # Dirichlet boundary, just as LinearElasticitySolver

        # how about kinematic energy, only for dynamic process vibration
        if self.transient_settings['transient']:
            vel = (u_current - u_prev) /dt
            Pi += 0.5*vel*vel*dx  # not yet tested code!
        
        if self.body_source:
            Pi -= dot(self.body_source, u_current)*dx

        ds= Measure("ds", subdomain_data=self.boundary_facets)  # if later marking updating in this ds?
        # hack solution: u_current as testfunction v
        bcs, integrals_F = self.update_boundary_conditions(time_iter_, u, u_current, ds)
        if time_iter_==0:
            plot(self.boundary_facets, title = "boundary facets colored by ID")
        # Assemble system, applying boundary conditions and extra items
        if len(integrals_F):
            for item in integrals_F: Pi -= item

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = derivative(Pi, u_current, v)
        self.J = derivative(F, u_current, u)
        return F, bcs

    def solve_form(self, F, u_, bcs):
        solve(F == 0, u_, bcs, J=self.J)
        return u_

