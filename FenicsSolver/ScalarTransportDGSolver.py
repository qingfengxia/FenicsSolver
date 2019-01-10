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

supported_scalers = {'temperature', 'electric_potential', 'species_concentration'}


from .ScalarTransportSolver import ScalarTransportSolver
from .SolverBase import SolverBase, SolverError
class ScalerTransportDGSolver(ScalarTransportSolver):
    """ share code as much as possible withe the CG version, 
    adapted from official tutorial:  2D, no source, no Neumann boundary
    https://github.com/FEniCS/dolfin/tree/master/demo/undocumented/dg-advection-diffusion
    """
    def __init__(self, s):
        ScalarTransportSolver.__init__(self, s)
        self.using_diffusion_form = True

    def generate_function_space(self, periodic_boundary):
        self.is_mixed_function_space = False
        if periodic_boundary:
            self.function_space_CG = FunctionSpace(self.mesh, "CG", self.settings['fe_degree'], constrained_domain=periodic_boundary)
            self.function_space = FunctionSpace(self.mesh, "DG", self.settings['fe_degree'], constrained_domain=periodic_boundary)
            self.vector_function_space = VectorFunctionSpace(self.mesh, 'CG', self.settings['fe_degree']+1, constrained_domain=periodic_boundary)
            # the group and degree of the FE element.
        else:
            self.function_space_CG = FunctionSpace(self.mesh, "CG", self.settings['fe_degree'])
            self.function_space = FunctionSpace(self.mesh, "DG", self.settings['fe_degree'])
            self.vector_function_space = VectorFunctionSpace(self.mesh, 'CG', self.settings['fe_degree']+1)

    def get_convective_velocity_function(self, convective_velocity):
        # fixme: rename !
        #vel = self.translate_value(convective_velocity, self.vector_function_space)
        vel = convective_velocity
        #vel = Constant((1.0, 1.0))
        #print("vel.ufl_shape", vel.ufl_shape)
        return vel

    def generate_form(self, time_iter_, T, Tq, T_current, T_prev):
        parameters["ghost_mode"] = "shared_facet"
        # T, Tq can be shared between time steps, form is unified diffussion coefficient
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)  # cell size
        h_avg = (h('+') + h('-'))/2
        # Penalty term

        dx= Measure("dx", subdomain_data=self.subdomains)  # subdomain (MeshFunction) does not distinguish DG CG?
        ds= Measure("ds", subdomain_data=self.boundary_facets)

        conductivity = self.conductivity() # constant, experssion or tensor
        capacity = self.capacity()  # density * specific capacity -> volumetrical capacity
        diffusivity = self.diffusivity()  # diffusivity
        #print(conductivity, capacity, diffusivity)

        bcs, integrals_N = self.update_boundary_conditions(time_iter_, T, Tq, ds)

        def F_convective():
            velocity = self.get_convective_velocity_function(self.convective_velocity)
            vel_n = (dot(velocity, n) + abs(dot(velocity, n)))/2.0
            
            if False:
                #http://www.karlin.mff.cuni.cz/~hron/fenics-tutorial/discontinuous_galerkin/doc.html
                # the only difference between official DG advection tutorial is `alpha/avg(h)`
                Pe= 1.0/diffusivity
                alpha = 1
                theta = 0.5

                def a(u,v) :
                    # Bilinear form
                    a_int = dot(grad(v), (1.0/Pe)*grad(u) - velocity*u)*dx
                    
                    a_fac = (1.0/Pe)*(alpha/avg(h))*dot(jump(u, n), jump(v, n))*dS \
                            - (1.0/Pe)*dot(avg(grad(u)), jump(v, n))*dS \
                            - (1.0/Pe)*dot(jump(u, n), avg(grad(v)))*dS
                    
                    a_vel = dot(jump(v), vel_n('+')*u('+') - vel_n('-')*u('-') )*dS  + dot(v, vel_n*u)*ds
                    
                    a = a_int + a_fac + a_vel
                    return a

                if self.transient_settings['transient']:
                    # Define variational forms
                    a0=a(T_prev,Tq)
                    a1=a(T,Tq)

                    F = (1/dt)*inner(T, Tq)*dx - (1/dt)*inner(T_prev,Tq)*dx + theta*a1 + (1-theta)*a0
                else:
                    F = theta*a(T,Tq) + (1-theta)*a(T_prev,Tq)
                F = F * Constant(capacity)

            else:
                if self.dimension == 2:
                    alpha = Constant(5.0)  # default 5 for 2D, but it needs to be higher for 3D case
                else:
                    alpha = Constant(500)
                v = Tq
                phi = T
                kappa = diffusivity
                # ( dot(v, n) + |dot(v, n)| )/2.0

                # Bilinear form
                a_int = dot(grad(v), kappa*grad(phi) - velocity*phi)*dx

                a_fac = kappa*(alpha/h('+'))*dot(jump(v, n), jump(phi, n))*dS \
                      - kappa*dot(avg(grad(v)), jump(phi, n))*dS \
                      - kappa*dot(jump(v, n), avg(grad(phi)))*dS

                #`Numerical simulations of advection-dominated scalar mixing with applications to spinal CSF flow and drug transport`
                a_vel = dot(jump(v), vel_n('+')*phi('+') - vel_n('-')*phi('-') )*dS  + dot(v, vel_n*phi)*ds

                F = (a_int + a_fac + a_vel) * Constant(capacity)

            if integrals_N:
                print("integrals_N", integrals_N)
                F -= sum(integrals_N)  # FIXME: DG may need distinct newmann boundary flux
            # Linear form
            if self.body_source:
                F -= Tq * self.body_source * dx
            return F

        if not hasattr(self, 'convective_velocity'):  # if velocity is not directly assigned to the solver
            if 'convective_velocity' in self.settings and self.settings['convective_velocity']:
                self.convective_velocity = self.settings['convective_velocity']
            else:
                self.convective_velocity = None

        if self.convective_velocity:  # convective heat conduction
            F = F_convective()
            print('Discrete Galerkin method solves only advection-diffusion equation')
        else:
            F = None
            raise SolverError('Error: Discrete Galerkin method should be used with advection velocity')

        return F, bcs

    def solve_form(self, F, T_current, bcs):
        if False:
            self.solve_linear_problem(F, T_current, bcs)
        else:
            problem = LinearVariationalProblem(lhs(F), rhs(F), T_current, bcs)
            solver = LinearVariationalSolver(problem)
            solver.solve()
            """
            a, L = lhs(F), rhs(F)
            print(a, L)

            A = PETScMatrix()
            assemble(a, tensor=A)
            b = assemble(L)

            #for bc in bcs:              bc.apply(A, b)
            # Set up boundary condition (apply strong BCs)
            class DirichletBoundary(SubDomain):
                def inside(self, x, on_boundary):
                    return  on_boundary
            g = Expression("300", degree=1)
            bc = DirichletBC(self.function_space, g, DirichletBoundary(), "geometric")
            bc.apply(A, b)
            #bc.apply(A)  #error here:

            # Solve system
            solve(A, T_current.vector(), b)
            """
        return T_current

    def solve(self):
        _result = self.solve_transient()
        # Project solution to a continuous function space
        self.result = project(_result, V=self.function_space_CG)
        return self.result