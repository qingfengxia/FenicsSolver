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

from dolfin import *
from .NonlinearElasticitySolver import NonlinearElasticitySolver
from .SolverBase import SolverBase, SolverError

class LargeDeformationSolver(NonlinearElasticitySolver):
    """ 
    adapted from: http://www.karlin.mff.cuni.cz/~blechta/fenics-tutorial/elasticity/doc.html
    velocity is not vertex velocity! 
    kinematic energy is not added
    """
    def __init__(self, case_settings):
        NonlinearElasticitySolver.__init__(self, case_settings)
        
        case_settings['vector_name'] = 'displacement'
        # Use UFLACS to speed-up assembly and limit quadrature degree
        parameters['form_compiler']['representation'] = 'uflacs'
        parameters['form_compiler']['optimize'] = True
        parameters['form_compiler']['quadrature_degree'] = 4

    def generate_function_space(self, periodic_boundary):
        self.is_mixed_function_space = True
        print('self.is_mixed_function_space in the solver', self.is_mixed_function_space)
        V = VectorElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.settings['fe_degree']) 
        Q = FiniteElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.settings['fe_degree'])
        mixed_element = MixedElement([V, V, Q])  # displacement, velocity, pressure

        if periodic_boundary:
            self.function_space = FunctionSpace(self.mesh, mixed_element, constrained_domain=periodic_boundary)
        else:
            self.function_space = FunctionSpace(self.mesh, mixed_element)

    '''
    def get_initial_field(self):
        # assume:  all constant, velocity is a tupe of constant
        #_initial_values = []
        #_initial_values.append(self.dimension*(0.0,))
        #_initial_values.append(self.dimension*(0.0,))  # self.initial_values['velocity']
        # _initial_values.append(0.0)
        _initial_values = (self.dimension*2+1)*(0.0,)
        _expr = tuple([str(v) for v in _initial_values])
        print(_expr)
        _initial_values_expr = Expression( _expr, degree = self.settings['fe_degree'])
        up0 = interpolate(_initial_values_expr, self.function_space)
        return up0
    '''

    def get_flux(self, u, mag_vector): 
        F = Identity(self.dimension) + grad(u)
        print("mag_vector", mag_vector)
        return det(F)*dot(inv(F).T, mag_vector)

    def generate_form(self, time_iter_, w_trial, w_test, w_current, w_prev):
        
        dx= Measure("dx", subdomain_data=self.subdomains) # in case material property depends on subdomains
        #material property
        E = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = Constant(E/(2.0*(1.0 + nu)))  # shear modulus

        (u, v, p) = split(w_current)  # displacement, velocity, pressure
        (u0, v0, p0) = split(w_prev)
        (_u, _v, _p) = split(w_test)

        I = Identity(self.dimension)
    
        # Define constitutive law for large deformation, coordinates in deformed configuration
        def stress(u, p):
            """Define constitutive law for large deformation, coordinates in deformed configuration
            Returns 1st Piola-Kirchhoff stress and (local) mass balance for given u, p.
            see: """

            F = I + grad(u)  # deformation gradient
            J = det(F)  
            B = F * F.T  # left cauchy green deformation tensor, unsymmetric tensor,  while cauchy stress is symmetric
            T = -p*I + mu*(B-I) # Cauchy stress
            S = J*T*inv(F).T # 1st Piola-Kirchhoff stress
            if nu == 0.5:
                # Incompressible material, singularity problem 
                pp = J-1.0  # volumetric change
            else:
                # Compressible
                lmbd = Constant(E*nu/((1.0 + nu)*(1.0 - 2.0*nu)))  # volumetric modulus
                pp = 1.0/lmbd*p + (J*J-1.0)  # 
            return S, pp
    
        if self.transient_settings['transient']:
            dt = self.get_time_step(time_iter_)
            q = 0.5  # time fwd scheme 0.5: crank-niklas
        else:
            raise SolverError("large deformation solver must be solved in a transient way")
        
        # Balance of momentum, using velocity test function here
        S, pp = stress(u, p)
        S0, pp0 = stress(u0, p0)
        F1 = (1.0/dt)*inner(u-u0, _u)*dx \
           - ( q*inner(v, _u)*dx + (1.0-q)*inner(v0, _u)*dx )  # q is the time stepping constant
        F2a = inner(S, grad(_v))*dx + pp*_p*dx
        F2b = inner(S0, grad(_v))*dx + pp0*_p*dx
        F2 = (1.0/dt)*inner(v-v0, _v)*dx + q*F2a + (1.0-q)*F2b

        F = F1 + F2
        ds= Measure("ds", subdomain_data=self.boundary_facets)  # if later marking updating in this ds?
        bcs, integrals_F = self.update_boundary_conditions(time_iter_, u, _v, ds)
        if time_iter_==0:
            plot(self.boundary_facets, title = "boundary facets colored by ID")
            #interactive()

        if self.body_source:
            integrals_F.append(inner(self.body_source, _v)*dx )

        F += sum(integrals_F)  # FIXME: in original tutorial, boundary flux is added to F
        #to do: direction_vector if it is not a tuple,  constant of vector
        #nitsche for directly
        # def get_source_item():  generalised, done
        # Traction at boundary, stress or force?

        # Whole system and its Jacobian

        self.J = derivative(F, w_current)
        return F, bcs

    def solve_form(self, F, u_, bcs):
        solve(F == 0, u_, bcs, J=self.J,
                    solver_parameters={"newton_solver":{"linear_solver":"mumps","absolute_tolerance":1e-9,"relative_tolerance":1e-7}})
        return u_

    def displacement(self):
        if self.is_mixed_function_space:
            u_, v_, p_ = split(self.w_current)  
            return u_  # large deformation function space:  disp, vel, pressure

    def velocity(self):
        dt = self.get_time_step(self.current_step)
        if self.is_mixed_function_space:
            u_, v_, p_  = split(self.w_current)
            #return v_  # large deformation function space:  disp, vel, pressure, velocity not correct
            u0_, v0_, p0_  = split(self.w_prev)
            return (u_ - u0_)/Constant(dt)

    def plot_result(self):
        # Extract solution components and rename
        (u, v, p) = split(self.result)
        #v.rename("v", "velocity")
        #p.rename("p", "pressure")
        plot(u, mode="displacement", wireframe=True)
    ####################################

