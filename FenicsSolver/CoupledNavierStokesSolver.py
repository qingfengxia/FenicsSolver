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
#import math may cause error
import numbers
import numpy as np
import os.path

from dolfin import *

#Oasis: a high-level/high-performance open source Navier-Stokes solve

"""
TODO:
1. temperature
2. boundary translation
3. test parallel by MPI
"""

from .SolverBase import SolverBase
class CoupledNavierStokesSolver(SolverBase):
    """  incompressible and laminar only
    """
    def __init__(self, case_input):

        if 'solving_temperature' in case_input:
            self.solving_temperature = case_input['solving_temperature']
        else:
            self.solving_temperature = False
        SolverBase.__init__(self, case_input)

        # init and reference must be provided by case setup

        if self.solving_temperature:
            self.settings['mixed_variable'] = ('velocity', 'pressure', 'temperature')
        else:
            self.settings['mixed_variable'] = ('velocity', 'pressure')
        ## Define solver parameters, underreleax, tolerance, max_iter
        # solver_parameters

    def generate_function_space(self, periodic_boundary):
        self.vel_degree = self.settings['fe_degree'] + 1  # order 3 is working for 2D elbow testing
        self.pressure_degree = self.settings['fe_degree']
        self.is_mixed_function_space = True  # FIXME: how to detect it is mixed, if function_space is provided

        V = VectorElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.vel_degree)  # degree 2, must be higher than pressure
        Q = FiniteElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.pressure_degree)
        #T = FiniteElement("CG", self.mesh.ufl_cell(), 1)  # temperature subspace, or just use Q
        if self.solving_temperature:
            mixed_element = MixedElement([V, Q, Q])
        else:
            mixed_element = V * Q  # MixedFunctionSpace has been removed from 2016.2, this API works only for 2 sub
        if periodic_boundary:
            self.function_space = FunctionSpace(self.mesh, mixed_element, constrained_domain=periodic_boundary)
        else:
            self.function_space = FunctionSpace(self.mesh, mixed_element)

    def get_body_source(self):
        # FIXME: source term type, centrifugal force is possible
        if 'body_source' in self.settings and self.settings['body_source']:
            body_force = self.translate_value(self.settings['body_source'])
        else:  # default gravity
            if self.dimension == 2: 
                body_force = Constant((0, -9.8))
            else:
                body_force = Constant((0, 0, -9.8))
        return body_force

    def get_initial_field(self):
        # assume:  all constant, velocity is a tupe of constant
        # function assigner is another way, assign(m.sub(0), v0)
        print(self.initial_values)

        _initial_values = list(self.initial_values['velocity'])
        _initial_values.append(self.initial_values['pressure'])
        if self.solving_temperature:
            _initial_values.append(self.initial_values['temperature'])
            #self.function_space.ufl_element(), wht not working
        _initial_values_expr = Expression( tuple([str(v) for v in _initial_values]), degree = 1)
        up0 = interpolate(_initial_values_expr, self.function_space)
        return up0

    def viscous_stress(self, u, p):
        T_space = TensorFunctionSpace(self.function_space.mesh(), 'CG', self.vel_degree)
        sigma = project(self.viscosity()*(grad(u) + grad(u).T) - p*Identity(), T_space, 
            solver_type = 'mumps', form_compilder_parameters = {'cpp_optimize':  True, "representation": 'quadrature', "quadrature_degree": 2})
        return sigma

    def calc_drag_and_lift(self, u, p, drag_axis_index, lift_axis_index, boundary_index_list):
        # Compute force on cylinder
        T = self.viscous_stress(u, p)
        n = FacetNormal(self.mesh)
        if (boundary_index_list and len(boundary_index_list)):
            drag = -T[drag_axis_index,j]*n[j]*self.ds(boundary_index_list[0])
            lift = -T[lift_axis_index,j]*n[j]*self.ds(boundary_index_list[0])
            for _i in boundary_index_list[1:]:
                drag -= T[0,j]*n[j]*self.ds(_i)  #  index j means summnation
                lift -= T[1,j]*n[j]*self.ds(_i)
            drag = assemble(drag)
            lift = assemble(lift)
            return drag, lift
        else:
            raise SolverError('Error: boundary_index_list must be specified to calc drag and lift forces')

    def viscous_heat(self, u, p):
        # shear heating
        #V_space = VectorFunctionSpace(self.function_space.mesh())
        V_space = self.function_space.sub(0)
        return project(inner(self.viscosity()*(grad(u) + grad(u).T) - p*Identity(), u) , V_space,
            solver_type = 'mumps', form_compilder_parameters = {'cpp_optimize':  True, "representation": 'quadrature', "quadrature_degree": 2})

    def viscosity(self):
        # TODO material nonlinear is not yet implemented
        _nu = self.material['kinematic_viscosity']
        if isinstance(_nu, (Constant, numbers.Number)):
            nu = Constant(_nu)
        #else:  #if nu is nu is string or function
        #def viscosity_function(u, T, p):
        #   return Constant(1)*pow(p/Constant(self.reference_values['pressure']), 2)
        #   _nu = viscosity(u, p)
        return _nu  # nonlinear, nonNewtonian

    def generate_form(self, time_iter_, trial_function, test_function, up_current, up_prev):
        W = self.function_space
        ## boundary setup and update for each time step

        print("Updating boundary at time iter = {}".format(time_iter_))
        ds = Measure("ds", subdomain_data=self.boundary_facets)
        if time_iter_ == 0:
            plot(self.boundary_facets, title ="boundary colored by ID")  # diff color do visual diff boundary 

        # Define unknown and test function(s)

        ## weak form
        F = self.F_static(trial_function, test_function, up_current)
        if self.transient:  # temporal scheme
            F += self.F_transient(time_iter_, trial_function, test_function, up_current, up_prev)

        Dirichlet_bcs_up, F_bc = self.update_boundary_conditions(time_iter_, trial_function, test_function, ds)
        for it in F_bc:
            F += it
        return F, Dirichlet_bcs_up

    def F_static(self, trial_function, test_function, up_0):
        u, p = split(trial_function)
        v, q = split(test_function)
        u_0, p_0 = split(up_0)
        def epsilon(u):
            """Return the symmetric gradient."""
            return 0.5 * (grad(u) + grad(u).T)

        nu = self.viscosity()
        # Define Form for the static Stokes Coupled equation,
        F = nu * 2.0*inner(epsilon(u), epsilon(v))*dx \
            - p*div(v)*dx \
            + div(u)*q*dx
        if self.settings['body_source']: 
            F -= inner(self.get_body_source(), v)*dx  # dot() ?
        # Add convective term
        F += inner(dot(grad(u), u_0), v)*dx  # u_0 is the current value solved
        return F

    def F_transient(self, time_iter_, trial_function, test_function, up_0, up_prev):
        u, p = split(trial_function)
        v, q = split(test_function)
        u_0, p_0 = split(up_0)  # up_0 not in use
        u_prev, p_prev = split(up_prev)
        return (1 / self.get_time_step(time_iter_)) * inner(u - u_prev, v) * dx  # dot() ?

    def update_boundary_conditions(self, time_iter_, trial_function, test_function, ds):
        W = self.function_space
        n = FacetNormal(self.mesh)  # used in pressure force

        if self.solving_temperature:
            u, p, pT = split(trial_function)
            v, q, qT = split(test_function)
        else:
            u, p = split(trial_function)
            v, q = split(test_function)

        Dirichlet_bcs_up = []
        F_bc = []
        #Dirichlet_bcs_up = [DirichletBC(W.sub(0), Constant(0,0,0), boundary_facets, 0)] #default vel, should be set before call this
        for key, boundary in self.boundary_conditions.items():
            #print(boundary)
            if isinstance(boundary['values'], list):
                bc_values = boundary['values']
            else:
                bc_values = boundary['values'].values()
            for bc in bc_values:  # a list of boundary values or dict
                if bc['variable'] == 'velocity':
                    print(bc['value'])
                    bvalue = self.translate_value(bc['value'])
                    '''
                    if hasattr(bc['value'], '__len__') and len(bc['value']) == self.dimension:
                        bvalue = self.translate_value(bc['value'])
                    else:  # scaler
                        #bvalue = self.translate_value(bc['value'])*n
                        raise TypeError('FacetNormal can not been used in Dirichlet boundary')
                    '''
                    if bc['type'] == 'Dirichlet':
                        Dirichlet_bcs_up.append(DirichletBC(W.sub(0), bvalue, self.boundary_facets, boundary['boundary_id']) )
                        print("found velocity boundary for id = {}".format(boundary['boundary_id']))
                    elif bc['type'] == 'Neumann':  # zero gradient, outflow
                        NotImplementedError('Neumann boundary for velocity is not implemented')
                    elif bc['type'] == 'symmetry' or bc['type'] == 'farfield': 
                        pass   # velocity gradient is zero, do nothing here, no normal stress, see [COMSOL Multiphysics Modeling Guide]
                    else:
                        print('velocity boundary type`{}` is not supported'.format(bc['type']))
                elif bc['variable'] == 'pressure':
                    bvalue = self.translate_value(bc['value'])  # self.get_boundary_value(bc, 'pressure')
                    if bc['type'] == 'Dirichlet':  # pressure  inlet or outlet
                        Dirichlet_bcs_up.append(DirichletBC(W.sub(1), bvalue, self.boundary_facets, boundary['boundary_id']) )
                        F_bc.append(inner(bvalue*n, v)*ds(boundary['boundary_id'])) # very import to make sure convergence
                        F_bc.append(-self.viscosity()*inner((grad(u) + grad(u).T)*n, v)*ds(boundary['boundary_id']))  #  pressure no viscous stress boundary
                        print("found pressure boundary for id = {}".format(boundary['boundary_id']))
                    elif bc['type'] == 'symmetry':
                        # normal stress project to tangital directions:  t*(-pI + viscous_force)*n
                        F_bc.append(-self.viscosity()*inner((grad(u) + grad(u).T)*n, v)*ds(boundary['boundary_id']))
                    elif bc['type'] == 'farfield':   # 'open' to large volume is same with farfield
                        F_bc.append(-self.viscosity()*inner((grad(u) + grad(u).T)*n, v)*ds(boundary['boundary_id']))  #  pressure no viscous stress boundary
                    elif bc['type'] == 'Neumann':  # zero gradient
                        NotImplementedError('Neumann boundary for pressure is not implemented')
                    else:
                        print('pressure boundary type`{}` is not supported thus ignored'.format(bc['type']))
                elif bc['variable'] == 'temperature' and self.solving_temperature:  # used by compressible NS solver
                    bvalue = self.translate_value(bc['value'])
                    if bc['type'] == 'Dirichlet':  # pressure  inlet or outlet
                        Dirichlet_bcs_up.append(DirichletBC(W.sub(2), bvalue, self.boundary_facets, boundary['boundary_id']) )
                        print("found temperature boundary for id = {}".format(boundary['boundary_id']))
                        """
                        elif bc['type'] == 'Neumann':  # zero gradient
                            F -=
                        elif bc['type'] == 'Robin':  # zero gradient
                            F -=
                        """
                    else:
                        print('temperature boundary type`{}` is not supported thus ignored'.format(bc['type']))
                else:
                    print('boundary value `{}` is not supported thus ignored'.format(bc['variable']))
        ## end of boundary setup
        return Dirichlet_bcs_up, F_bc

    def solve_nonlinear(self, time_iter_, up_0, up_prev):
        raise NotImplementedError(" nonlinear viscosity model is not implemented yet")
        """
        iter_ = 0
        max_iter = 50
        eps = 1.0
        tol = 1E-4

        timer_solver = Timer("TimerSolveNonlinearViscosity")
        timer_solver.start()
        while (iter_ < max_iter and eps > tol):
            F, Dirichlet_bcs_up = self.update_boundary_conditions(time_iter_, up_0, up_prev)
            up_0 = self.solve_static(F, up_0, Dirichlet_bcs_up)
            iter_ += 1
        ## end of Picard loop
        timer_solver.stop()
        print("*" * 10 + " end of nonlinear viscosity iteration" + "*" * 10)

        return up_0
        """

    def solve_static(self, F, up_, Dirichlet_bcs_up):
        # Solve stationary Navier-Stokes problem with Picard method
        # other methods may be more acurate and faster

        iter_ = 0
        max_iter = 50
        eps = 1.0
        tol = 1E-3
        under_relax_ratio = 0.7
        up_temp = Function(self.function_space)  # a temporal to save value in the Picard loop

        timer_solver = Timer("TimerSolveStatic")
        timer_solver.start()
        while (iter_ < max_iter and eps > tol):
            # solve the linear stokes flow to avoid up_s = 0

            up_temp.assign(up_)
            # other solving methods
            up_ = self.solve_linear_problem(F, up_, Dirichlet_bcs_up)
            #up_s = self.solve_amg(F, Dirichlet_bcs_up, up_s)  #  AMG is not working with mixed function space

            diff_up = up_.vector().array() - up_temp.vector().array()
            eps = np.linalg.norm(diff_up, ord=np.Inf)

            print("iter = {:d}; eps_up = {:e}; time elapsed = {}\n".format(iter_, eps, timer_solver.elapsed()))

            ## underreleax should be defined here, Courant number, 
            up_.vector()[:] = up_temp.vector().array() + diff_up * under_relax_ratio

            iter_ += 1
        ## end of Picard loop
        timer_solver.stop()
        print("*" * 10 + " end of Navier-Stokes equation iteration" + "*" * 10)

        return up_

    def solve(self):
        self.result = self.solve_transient()
        return self.result

    def plot(self):
        u,p= split(self.result)
        plot(u)  # error in plot() for version 2017.2 in Python3
        plot(p)
        if self.report_settings['plotting_interactive']:
            interactive()
