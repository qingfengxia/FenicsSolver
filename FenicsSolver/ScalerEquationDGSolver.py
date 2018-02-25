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


# this solver has not passed testing, all result is NAN

from __future__ import print_function, division
import math
import numpy as np

from dolfin import *

supported_scaler_equations = {'temperature', 'electric_potential', 'species_concentration'}


from .ScalerEquationSolver import ScalerEquationSolver
from .SolverBase import SolverBase, SolverError
class ScalerEquationDGSolver(ScalerEquationSolver):
    """ share code as much as possible withe the CG version, 
    adapted from official tutorial:  2D, no source, no Neumann boundary
    https://github.com/FEniCS/dolfin/tree/master/demo/undocumented/dg-advection-diffusion
    """
    def __init__(self, s):
        ScalerEquationSolver.__init__(self, s)
        self.using_diffusion_form = True

    def generate_function_space(self, periodic_boundary):
        if periodic_boundary:
            self.function_space_CG = FunctionSpace(self.mesh, "CG", self.degree, constrained_domain=periodic_boundary)
            self.function_space = FunctionSpace(self.mesh, "DG", self.degree, constrained_domain=periodic_boundary)
            self.vector_function_space = VectorFunctionSpace(self.mesh, 'CG', self.degree+1, constrained_domain=periodic_boundary)
            # the group and degree of the FE element.
        else:
            self.function_space_CG = FunctionSpace(self.mesh, "CG", self.degree)
            self.function_space = FunctionSpace(self.mesh, "DG", self.degree)
            self.vector_function_space = VectorFunctionSpace(self.mesh, 'CG', self.degree+1)

    def get_convective_velocity_function(self, convective_velocity):
        # fixme: rename !
        #vel = self.translate_value(convective_velocity, self.vector_function_space)
        vel = convective_velocity
        #vel = Constant((1.0, 1.0))
        #print("vel.ufl_shape", vel.ufl_shape)
        return vel

    def generate_form(self, time_iter_, T, Tq, T_current, T_prev):
        # T, Tq can be shared between time steps, form is unified diffussion coefficient
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)  # cell size
        h_avg = (h('+') + h('-'))/2
        # Penalty term
        alpha = Constant(5.0)

        dx= Measure("dx", subdomain_data=self.subdomains)  # 
        ds= Measure("ds", subdomain_data=self.boundary_facets)

        conductivity = self.conductivity() # constant, experssion or tensor
        capacity = self.capacity()  # density * specific capacity -> volumetrical capacity
        diffusivity = self.diffusivity()  # diffusivity

        bcs, integrals_N = self.update_boundary_conditions(time_iter_, T, Tq, ds)

        # poission equation, unified for all kind of variables
        def F_static(T, Tq):
            F =  inner( conductivity * grad(T), grad(Tq))*dx
            if integrals_N:
                F -= sum(integrals_N)
            if self.body_source:
                F -= Tq * self.body_source * dx
            return F

        def F_convective():
            velocity = self.get_convective_velocity_function(self.convective_velocity)
            if self.transient_settings['transient']:
                raise NotImplementedError()
            else:
                v = Tq
                phi = T
                kappa = diffusivity
                # ( dot(v, n) + |dot(v, n)| )/2.0
                vel_n = (dot(velocity, n) + abs(dot(velocity, n)))/2.0

                # Bilinear form
                a_int = dot(grad(v), kappa*grad(phi) - velocity*phi)*dx

                a_fac = kappa*(alpha/h('+'))*dot(jump(v, n), jump(phi, n))*dS \
                      - kappa*dot(avg(grad(v)), jump(phi, n))*dS \
                      - kappa*dot(jump(v, n), avg(grad(phi)))*dS

                a_vel = dot(jump(v), vel_n('+')*phi('+') - vel_n('-')*phi('-') )*dS  + dot(v, vel_n*phi)*ds

                F = a_int + a_fac + a_vel
                if integrals_N:
                    print("integrals_N", integrals_N)
                    F -= sum(integrals_N)  #DG may need distinct newmann boundary flux
                # Linear form
                if self.body_source:
                    F -= v * self.body_source * dx
            #return F
            return F

        if self.convective_velocity:  # convective heat conduction
            F = F_convective()
        else:
            F = F_static(T, Tq)
            print('Discrete Galerkin method only solver advection-diffusion scaler equation, \
            use CG for diffusion only equation')

        return F, bcs

    def solve_static(self, F, T_current, bcs):
        # Project solution to a continuous function space
        a, L = lhs(F), rhs(F)
        A = assemble(a)
        b = assemble(L)
        for bc in bcs:
            bc.apply(A, b)

        # Solve system
        solve(A, T_current.vector(), b)
        up = project(T_current, V=self.function_space_CG)
        return up