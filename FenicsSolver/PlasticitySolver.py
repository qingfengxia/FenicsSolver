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

class PlasticitySolver(LinearElasticitySolver):
    """ share code as much as possible withe the linear version, especially boundary settings
    only simple plasticity models are supported : 
    Fenics's SolidMechanics module has complicate and efficient plasticity models in cpp
    see <>
    """
    supported_plasticity_models = []
    def __init__(self, s):
        LinearElasticitySolver.__init__(self, s)
        #plasticity special setting
        if 'plasticity_model' in self.settings:
            self.plasticity = self.settings['plasticity_model']
        else:
            raise SolverError('plasticity_model is not provided in settings')

        # Optimization options for the form compiler
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["representation"] = "uflacs"
        #
        parameters["form_compiler"]["cpp_optimize"]= True
        parameters["form_compiler"]["quadrature_degree"] = 4

    def get_initial_field(self):
        V = self.function_space
        u0 = Function(V)
        self.StressFunctionSpace = TensorFunctionSpace(self.mesh,"DG",1)
        self.DGFunctionSpace = FunctionSpace(self.mesh,"DG",1)

        self.Sigma_Y0=Function(self.DGFunctionSpace, name='yield')# local yield stress
        self.Sigma_Y=Function(self.DGFunctionSpace, name='local')# local yield stress
        self.Sigma0=Function(self.StressFunctionSpace)
        #self.Sigma is calc in generate_form

        self.Sigma_Y0.vector()[:]=200# initilize
        self.Sigma_Y.vector()[:]=200# initilize
        return u0

    def get_form_F(self, time_iter_, u_trial, u_test, u_current, u_prev):
        # linear elasticity, exxcept deps
        elasticity = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = elasticity/(2.0*(1.0 + nu))
        lmbda = elasticity*nu/((1.0 + nu)*(1.0 - 2.0*nu))

        I = Identity(self.dimension)             # Identity tensor
        deps=sym(grad(u_current - u_prev))  #total deformation=deps_e+deps_p, only diff
        dsigma=2*mu*deps+lmbda*tr(deps)*I
        Sigma_tr=self.Sigma0+dsigma

        F=inner(Sigma_tr, grad(u_test))*dx
        # apply body source and boundary condition (point, surface source)
        if self.body_source:
            F -= inner(self.body_source, u_test)*dx
        ds= Measure("ds", subdomain_data=self.boundary_facets)  # if later marking updating in this ds?
        bcs, integrals_F = self.update_boundary_conditions(time_iter_, u_trial, u_test, ds)
        if time_iter_==0:
            plot(self.boundary_facets, title = "boundary facets colored by ID")
        # Assemble system, applying boundary conditions and extra items
        if len(integrals_F):
            for item in integrals_F: F -= item
        return F, bcs

    def solve_initial_field(self, u_trial, u_test, u_current, u_prev):
        F, bcs = self.get_form_F(0, u_trial, u_test, u_current, u_prev)
        J=derivative(F, u_current, u_trial)

        A, rhs = assemble_system(J, -F, bcs)  # why minus -F ?
        solve(A, u_current.vector(), rhs, "mumps")
        return u_current

    def solve_current_step(self, trial_function, test_function, w_current, w_prev):
        if self.current_step == 0:  # PlasticitySolver.py
            if hasattr(self, 'solve_initial_field'):
                self.solve_initial_field(trial_function, test_function, w_current, w_prev)

        F, Dirichlet_bcs = self.generate_form(self.current_step, trial_function, test_function, w_current, w_prev)
        #up_current = self.solve_static(F, up_current, Dirichlet_bcs_up)  # solve for each time step, up_prev tis not needed
        solve(F==0, w_current, Dirichlet_bcs, J=self.J, 
                solver_parameters={"newton_solver":{"linear_solver":"mumps","absolute_tolerance":1e-7,"relative_tolerance":1e-7}})

        w_prev.assign(w_current)
        self.Sigma_Y0.vector()[:]=project(self.Sigma_Y, self.DGFunctionSpace, solver_type='gmres').vector()
        self.Sigma0.vector()[:]=project(self.Sigma, self.StressFunctionSpace, solver_type="gmres").vector()

    def generate_form(self, time_iter_, u_trial, u_test, u_current, u_prev):
        elasticity = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = elasticity/(2.0*(1.0 + nu))
        lmbda = elasticity*nu/((1.0 + nu)*(1.0 - 2.0*nu))
        E_t = elasticity * 0.3 # ??
        H=E_t/(1.0-E_t/elasticity)

        I = Identity(self.dimension)             # Identity tensor
        deps=sym(grad(u_current - u_prev)) #total deformation=deps_e+deps_p
        dsigma=2*mu*deps+lmbda*tr(deps)*I
        Sigma_tr=self.Sigma0+dsigma

        Shear_tr=Sigma_tr-1./3*tr(Sigma_tr)*Identity(3)
        Sigma_eq_tr=sqrt(1.5*inner(Shear_tr, Shear_tr))
        Shear_norm_tr=sqrt(inner(Shear_tr, Shear_tr))
        #F_tr=Shear_norm_tr-(2./3.)**0.5*Sigma_Y0
        F_tr=Sigma_eq_tr - self.Sigma_Y0

        deps_eq=conditional(F_tr>=0,F_tr/(H+3*mu), 0.)# when F_tr<0 return 0
        #Sigma=Sigma_tr-2*mu*gamma_dot*Shear_tr/Shear_norm_tr# radial return mapping
        Sigma=Sigma_tr-3*mu*deps_eq*Shear_tr/Sigma_eq_tr# radial return mapping
        Shear=Sigma-1./3*tr(Sigma)*Identity(3)
        Sigma_eq_tr=sqrt(1.5*inner(Shear, Shear))
        # why comment out? sigmaY is needed 
        gamma_dot=conditional(F_tr>=0,F_tr/(2./3*H+2*mu), 0.)# when F_tr<0 return 0
        Sigma_Y=self.Sigma_Y0+(2./3)**0.5*H*gamma_dot  # sigma is updated in original example

        F=inner(Sigma, grad(u_test))*dx
        # apply body source and boundary condition (point, surface source)
        if self.body_source:
            F -= dot(self.body_source, u_test)*dx
        ds= Measure("ds", subdomain_data=self.boundary_facets)  # if later marking updating in this ds?
        bcs, integrals_F = self.update_boundary_conditions(time_iter_, u_trial, u_test, ds)
        if time_iter_==0:
            plot(self.boundary_facets, title = "boundary facets colored by ID")
        # Assemble system, applying boundary conditions and extra items
        if len(integrals_F):
            for item in integrals_F: F -= item
        
        J=derivative(F, u_current, u_trial)
        self.J = J
        self.Sigma_Y = Sigma_Y
        self.Sigma = Sigma
        return F, bcs

