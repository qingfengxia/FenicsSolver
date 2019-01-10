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
import copy

from dolfin import *


"""
Feature:  coupled velocity and pressure laminar flow with G2 stabilisaton

General Galerkin (G2) stabilisaton (reference paper: ), but it still does not work for Re>10.
Turbulent flow will not be implemented: use third-party solvers like Oasis: a high-level/high-performance open source Navier-Stokes solver
OpenFOAM solver.

TODO:
1. temperature induced natural convection, need solve thermal, constant thermal expansion coeffi
    see paper:

2. noninertial frame of reference
linear accelation can be merged to body source, just as body
    + ALE:  mesh moving velocity, for FSI solver
    + SRF: for centrifugal pump, add item of centrifugal and coriolis forces
    + MRF:  for turbine blade and stator, should be done in a coupling way

reference_frame_settings = {'type': 'ALE',  'mesh_velocity': vel}
reference_frame_settings = {'type': 'SRF',  'center_point': (0, 0, 0), 'omega': -1}
                                            omega: angular velocity  rad/s, minus sign mean ?? direction
                                            omega*omega*x_vector + cross(2*omega, u_vector)
                                            stab for coriolis force item?

3. nonlinear viscosity model, nu(T), nu(U, p, T), curerntly diverging !
"""

from .SolverBase import SolverBase
class CoupledNavierStokesSolver(SolverBase):
    """  incompressible and laminar flow only with G2 stabilisaton
    """
    def __init__(self, case_input):

        if 'solving_temperature' in case_input:
            self.solving_temperature = case_input['solving_temperature']
        else:
            self.solving_temperature = False
        SolverBase.__init__(self, case_input)
        self.compressible = False
        # init and reference must be provided by case setup

        self.using_nonlinear_solver = True
        #self.settings['solver_name'] = "CoupledNavierStokesSolver"
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
        self._update_function_space(periodic_boundary)

    def _update_function_space(self, periodic_boundary=None):
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
        self.velocity_subfunction_space = self.function_space.sub(0)

    def update_solver_function_space(self, periodic_boundary=None):
        #used by FSI solver, after mesh moving
        self._update_function_space(periodic_boundary)

        self.trial_function = TrialFunction(self.function_space)
        self.test_function = TestFunction(self.function_space)
        # Define functions for new function space
        _w_current =  Function(self.function_space)
        _w_current.vector()[:] = self.w_current.vector()[:]  # assuming no topo change
        self.w_current = _w_current
        _w_prev = Function(self.function_space)  # on previous time-step mesh
        _w_prev.vector()[:] = self.w_prev.vector()[:]  # assuming no topo change
        self.w_prev = _w_prev  # 

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
        # assume: velocity is a tupe of constant, or string expression, or a function of the mixed functionspace
        print("self.initial_values = ", self.initial_values)

        if isinstance(self.initial_values, (Function,)):
            try:
                up0 = Function(self.initial_values)  # same mesh and function space
            except:
                up0 = project(self.initial_values, self.function_space)
            return up0

        _initial_values = list(self.initial_values['velocity'])
        _initial_values.append(self.initial_values['pressure'])
        if self.solving_temperature:
            _initial_values.append(self.initial_values['temperature'])
            #self.function_space.ufl_element(), wht not working
        _initial_values_expr = Expression( tuple([str(v) for v in _initial_values]), degree = self.settings['fe_degree'])
        up0 = interpolate(_initial_values_expr, self.function_space)
        return up0

    def viscous_stress(self, up, T_space):
        u, p = split(up)  # TODO: not general split
        if not T_space:
            T_space = TensorFunctionSpace(self.function_space.mesh(), 'CG', 1)
        sigma = project(self.viscosity()*(grad(u) + grad(u).T) - p*Identity(self.dimension), T_space, \
            solver_type = 'mumps')
        return sigma

    def boundary_traction(self, up, target_space = None):
        # https://www.openfoam.com/documentation/cpp-guide/html/classFoam_1_1functionObjects_1_1externalCoupled.html#a1063d7a675858ee0e647e36abbefe463
        sigma = self.viscous_stress(up)  # already included pressure
        n = FacetNormal(self.mesh)
        # stress_normal = dot(n, dot(sigma, n))  #scalar, pressure
        # (tangential stress vector) = dot(sigma, n) - stress_normal*n
        #traction = dot(sigma, n)  # may need as_vector, can not project?
        traction = sigma[i,j]*n[j]  # ufl.log.UFLException: Shapes do not match
        #traction = Constant((1e2*self.current_step, 0))  # TODO: tmp bypass for test
        #print('before traction project ...')
        if target_space:
            traction = interpolate(traction, target_space)  # project does not work
        return traction

    def calc_drag_and_lift(self, up, drag_axis_index, lift_axis_index, boundary_index_list):
        # Compute force on cylinder,  axis_index: 0 for x-axis, 1 for y_axis
        T = self.viscous_stress(up)
        n = FacetNormal(self.mesh)
        if (boundary_index_list and len(boundary_index_list)):
            drag = -T[drag_axis_index,j]*n[j]*self.ds(boundary_index_list[0])
            lift = -T[lift_axis_index,j]*n[j]*self.ds(boundary_index_list[0])
            for _i in boundary_index_list[1:]:
                drag -= T[drag_axis_index,j]*n[j]*self.ds(_i)  #  index j means summnation
                lift -= T[lift_axis_index,j]*n[j]*self.ds(_i)
            drag = assemble(drag)
            lift = assemble(lift)
            return drag, lift
        else:
            raise SolverError('Error: boundary_index_list must be specified to calc drag and lift forces')

    def viscous_heat(self, u, p):
        # shear heating power,  FIXME: not tested code
        V_space = self.function_space.sub(0)
        return project(inner(self.viscosity()*(grad(u) + grad(u).T) - p*Identity(self.dimension), grad(u)) , V_space,
            solver_type = 'mumps', form_compilder_parameters = \
                                {'cpp_optimize':  True, "representation": 'quadrature', "quadrature_degree": 2})

    def viscosity(self, current_w=None):
        # TODO: define rheology_model = {}, strain_rate dep viscosity is not implemented
        if 'Newtonian' in self.material and (not self.material['Newtonian']):
            if not current_w:
                current_w = self.w_current  # newton nonliner solver will replace self.trial_function to self.w_current
            if self.solving_temperature:
                u,p,T = split(current_w)
                _nu = self.material['kinematic_viscosity']
                _nu = _nu * (1 + (p/self.reference_values['pressure']) * 0.1)
                _nu = _nu * (1 - (T/self.reference_values['temperature']) * 0.2)
            else:
                u,p = split(current_w)
                _nu = self.material['kinematic_viscosity']
                _nu = _nu * pow(p/self.reference_values['pressure'], 0.1)
        else:
            _nu = self.material['kinematic_viscosity']
            #if isinstance(_nu, (Constant, numbers.Number)):
            #    nu = Constant(_nu)
            #else:  #if nu is nu is string or function
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
        if self.transient_settings['transient']:  # temporal scheme
            F = self.F_transient(time_iter_, trial_function, test_function, up_current, up_prev)
        else:
            F = self.F_static(trial_function, test_function, up_current)
        # added boundary to F_static() or here to make Crank-Nicolson temporal scheme
        Dirichlet_bcs_up, F_bc = self.update_boundary_conditions(time_iter_, trial_function, test_function, ds)
        for it in F_bc:
            F += it

        if self.solving_temperature:  # test not passed!
            F_T, T_bc = self.generate_thermal_form(time_iter_, trial_function, test_function, up_current, up_prev)
            F += F_T
            Dirichlet_bcs_up += T_bc

        if self.using_nonlinear_solver:
            F = action(F, up_current)
            self.J = derivative(F, up_current, trial_function)  #

        return F, Dirichlet_bcs_up

    def generate_thermal_form(self, time_iter_, trial_function, test_function, up_current, up_prev):
            # temperature related items for the coupled vel, pressure and temperature form
            assert not self.compressible
            u, p, T = split(trial_function)
            v, q, Tq = split(test_function)
            u_current, p_current, T_current = split(up_current)
            u_prev, p_prev, T_prev = split(up_prev)

            #translate_setting, set the FunctionSpace
            Tsettings = copy.copy(self.settings)
            Tsettings['scalar_name'] = 'temperature'
            Tsettings['mesh'] = None
            Tsettings['function_space'] =  self.function_space.sub(2)
            #Tsettings['convective_velocity'] = u_current  # can not use u_trial_function
            #Tsettings['advection_settings'] = {'stabilization_method': 'SPUG', 'Pe': 1.0/(15.0/(4200*1000))}
            # it is possible to use SPUG (not rotating velocity),  with and without body source
            Tsettings['advection_settings'] = {'stabilization_method': 'IP', 'alpha': 0.1}
            #Tsettings['body_source'] = # incomplate form?
            import pprint
            pprint.pprint(Tsettings)
            from FenicsSolver import  ScalarTransportSolver
            Tsolver = ScalarTransportSolver.ScalarTransportSolver(Tsettings)
            #Tsover.is_mixed_function_space = False
            
            #print('type(u_current)', u_current, type(u_current))  # ufl.tensors.ListTensor
            #print('type(u)', u, type(u))
            Tsolver.convective_velocity = u_current
            #ds should be passed to Tsolver? but they are actually same
            #convection stab can be a problem!
            F_T, T_bc = Tsolver.generate_form(time_iter_, T, Tq, T_current, T_prev)
            #inner() ufl.log.UFLException: Shapes do not match: <Sum id=140066973439888> and <ListTensor id=140066973395664>.
            tau = self.viscosity()*(grad(u) + grad(u).T) 
            #sigma = tau- p*Identity(self.dimension)  # u_current for both nonlinear and picard loop
            epsdot = 0.5 * (grad(u) + grad(u).T)
            viscous_heating = inner(epsdot, tau)  # method 1
            #viscous_heating =  2 * self.viscosity(up_current) * tr(dot(epsdot, epsdot))
            print('type of viscous heating:', type(viscous_heating))
            ## sigma * u:  heat flux,  sigma * grad(u) heat source
            #F_T -= viscous_heating *Tq*dx  # need sum to scalar!
            return F_T, T_bc

    def F_static(self, trial_function, test_function, up_0):
        if self.solving_temperature:
            u, p, T = split(trial_function)
            v, q, Tq = split(test_function)
            u_0, p_0, T_0 = split(up_0)
        else:
            u, p = split(trial_function)
            v, q = split(test_function)
            u_0, p_0 = split(up_0)
        def epsilon(u):
            """Return the symmetric gradient."""
            return 0.5 * (grad(u) + grad(u).T)
        '''
        if self.using_nonlinear_solver:
            advection_velocity = u
            nu = self.viscosity(trial_function)
        else:  # using Picard loop    FIXME: crank-nicolis scheme 
        '''
        advection_velocity = u_0 # u_0 is the current value to be solved for steady case
        nu = self.viscosity(up_0) # 
        mu = nu*self.material['density']
        rho = self.material['density']
        #plot(nu, title = 'viscosity')  # all zero
        #import matplotlib.pyplot as plt
        #plt.show()

        # Define Form for the static Stokes Coupled equation,  divided by rho on both sides
        F = nu * 2.0*inner(epsilon(u), epsilon(v))*dx \
            - (p/rho)*div(v)*dx \
            + div(u)*(q)*dx  # comes from incomperssible flow equation, diveded by rho?
        if self.settings['body_source']:   # just gravity, without * rho
            F -= inner(self.get_body_source(), v)*dx

        if 'reference_frame_settings' in self.settings:
            rfs = self.settings['reference_frame_settings']
            if rfs['type'] == 'ALE':  # also used in FSI mesh moving
                advection_velocity -= self.translate_value(rfs['mesh_velocity'])
                # treated mesh_velocity as part of advection, so it can be stabilized
            #elif rfs['type'] == 'SRF':
            #    pass
            else:
                raise SolverError('reference_frame_settings type `{}` is not supported'.format(rfs['type']))

        # Add advective term, technically, convection means
        F += inner(dot(grad(u), advection_velocity), v)*dx

        if 'advection_settings' in self.settings:
            ads = self.settings['advection_settings']  # a very big panelty factor can stabalize, but leading to diffusion error
        else:
            ads = {'stabilization_method': None}  # default none

        if ads['stabilization_method'] and ads['stabilization_method'] == 'G2':
            """
            ref:
            dt is uniform time step, and u^ ; T^ are values of velocity and temperature from previous time
            delta1 and delta2 are stablisation parameter defined in DG
            Johan Hoffman and Claes Johnson. Computational Turbulent Incompressible Flow,
            volume 4 of Applied Mathematics: Body and Soul. Springer, 2007.
            URL http://dx.doi.org/10.1007/978-3-540-46533-1.
            """
            h = 2*Circumradius(self.mesh)  # cell size
            if ads['Re']<=1:
                delta1 = ads['kappa1'] * h*h
                delta2 = ads['kappa2'] * h*h
            else:  # convection dominant, test_f<trial_f*h
                U0_square = dot(advection_velocity, advection_velocity)
                if self.transient:
                    dt = self.get_time_step(time_iter_)
                    delta1 = ads['kappa1'] /2.0 * 1.0/sqrt(1.0/(dt*dt) + 1.0/U0_square/h/h)
                else:
                    delta1 = ads['kappa1'] /2.0 * h/sqrt(U0_square)
                delta2 = ads['kappa2'] * h
            D_u =  delta1 * inner(dot(advection_velocity, grad(u)), dot(advection_velocity, grad(v)))*dx
            # D_u += delta2 * dot(grad(rho_0, grad(v)))  # D_s has the same item, why?
            # D_s =    density related item,  it should be ignored for incompressible NS equation
            F -= D_u

        return F

    def F_transient(self, time_iter_, trial_function, test_function, up_current, up_prev):
        if self.solving_temperature:
            u, p, T = split(trial_function)
            v, q, Tq = split(test_function)
            u_0, p_0, T_0 = split(up_current)
            u_prev, p_prev, T_prev = split(up_prev)
        else:
            u, p = split(trial_function)
            v, q = split(test_function)
            u_0, p_0 = split(up_current)
            u_prev, p_prev = split(up_prev)
        F = self.F_static(trial_function, test_function, up_current)
        #F_prev = self.F_static(prev, test_function, up_prev)  # should works for both Picard and Newton
        #TODO: it is backward Euler, not Crank-Nicolson (2nd , unconditionally stable for diffusion problem)
        return F + (1 / self.get_time_step(time_iter_)) * inner(u - u_prev, v) * dx

    def update_boundary_conditions(self, time_iter_, trial_function, test_function, ds):
        # shared by compressible and incompressible fluid solver
        #FIXME: for curved boundary, nitsche method is needed?
        W = self.function_space
        n = FacetNormal(self.mesh)  # used in pressure force

        # the sequence is different for velocity, pressure, temperature (to share code with compressible and incompressible solver)
        if self.solving_temperature:
            if not self.compressible:
                u, p, T = split(trial_function)
                v, q, Tq = split(test_function)
            else:
                p, u, T = split(trial_function)
                q, v, Tq = split(test_function)
        else:
            u, p = split(trial_function)
            v, q = split(test_function)

        nu = self.viscosity()

        if self.compressible:
            i_pressure, i_velocity, i_temperature =  0, 1, 2
        else:
            i_pressure, i_velocity, i_temperature =  1, 0, 2

        Dirichlet_bcs_up = []
        F_bc = []
        #Dirichlet_bcs_up = [DirichletBC(W.sub(0), Constant(0,0,0), boundary_facets, 0)] #default vel, should be set before call this
        for key, boundary in self.boundary_conditions.items():
            #print(boundary)
            if 'coupling' in boundary and boundary['coupling'] == 'FSI':
                if not 'values' in boundary:
                    #if values are not supplied, set to fixed wall condition
                    boundary['values'] = [{'variable': "velocity",'type': 'Dirichlet', 'value': self.dimension*(0.0,)}]

            if 'values' in boundary and isinstance(boundary['values'], list):
                bc_values = boundary['values']
            else:
                bc_values = boundary['values'].values()
            for bc in bc_values:  # a list of boundary values or dict
                if bc['variable'] == 'velocity':
                    print(bc['value'])
                    bvalue = self.translate_value(bc['value'])
                    '''  only velocity vector is acceptable, it must NOT be a magnitude scalar
                    if hasattr(bc['value'], '__len__') and len(bc['value']) == self.dimension:
                        bvalue = self.translate_value(bc['value'])
                    else:  # scalar
                        #bvalue = self.translate_value(bc['value'])*n
                        raise TypeError('FacetNormal can not been used in Dirichlet boundary')
                    '''
                    if bc['type'] == 'Dirichlet':
                        Dirichlet_bcs_up.append(DirichletBC(W.sub(i_velocity), bvalue, self.boundary_facets, boundary['boundary_id']) )
                        print("found velocity boundary for id = {}".format(boundary['boundary_id']))
                    elif bc['type'] == 'Neumann':  # zero gradient, outflow
                        NotImplementedError('Neumann boundary for velocity is not implemented')
                    elif bc['type'] == 'symmetry':
                        # normal stress project to tangital directions:  t*(-pI + viscous_force)*n
                        F_bc.append(inner(dot(u, n), v)*ds(boundary['boundary_id']))  # no velocity gradient across the boundary
                        F_bc.append(-nu*inner((grad(u) + grad(u).T)*n, v)*ds(boundary['boundary_id']))
                    elif bc['type'] == 'farfield':  # 'outflow'
                        F_bc.append(dot(grad(u), n)*v * ds(boundary['boundary_id']))
                        #velocity gradient is zero, do nothing here, no normal stress, see [COMSOL Multiphysics Modeling Guide]
                    else:
                        print('velocity boundary type`{}` is not supported'.format(bc['type']))
                elif bc['variable'] == 'pressure':
                    bvalue = self.translate_value(bc['value'])  # self.get_boundary_value(bc, 'pressure')
                    if bc['type'] == 'Dirichlet':  # pressure  inlet or outlet
                        Dirichlet_bcs_up.append(DirichletBC(W.sub(i_pressure), bvalue, self.boundary_facets, boundary['boundary_id']) )
                        F_bc.append(inner(bvalue*n, v)*ds(boundary['boundary_id'])) # very important to make sure convergence
                        F_bc.append(-nu*inner((grad(u) + grad(u).T)*n, v)*ds(boundary['boundary_id']))  #  pressure no viscous stress boundary
                        print("found pressure boundary for id = {}".format(boundary['boundary_id']))
                    elif bc['type'] == 'symmetry':
                        pass # already set in velocity, should be natural zero gradient for pressure
                    elif bc['type'] == 'farfield':   # 'open' to large volume is same with farfield
                        F_bc.append(-nu*inner((grad(u) + grad(u).T)*n, v)*ds(boundary['boundary_id']))  #  pressure no viscous stress boundary
                    elif bc['type'] == 'Neumann':  # zero gradient
                        NotImplementedError('Neumann boundary for pressure is not implemented')
                    else:
                        print('pressure boundary type`{}` is not supported thus ignored'.format(bc['type']))

                elif bc['variable'] == 'temperature':  # TODO: how to share code and boundary setup with ScalarTransportSolver
                    if self.compressible:  # used by compressible NS solver
                        bvalue = self.translate_value(bc['value'])
                        if bc['type'] == 'Dirichlet':
                            Dirichlet_bcs_up.append(DirichletBC(W.sub(i_temperature), bvalue, self.boundary_facets, boundary['boundary_id']) )
                            #print("found temperature boundary for id = {}".format(boundary['boundary_id']))
                        ''' # not yet implemented for those bc below
                        if bc['type'] == 'symmetry':
                            pass #F_bc.append(dot(grad(T), n)*Tq*ds(boundary['boundary_id']))
                        elif bc['type'] == 'Neumann' or bc['type'] =='fixedGradient':  # unit: K/m
                            g = self.translate_value(bc['value'])
                            integrals_N.append(capacity*g*Tq*ds(i))
                        elif bc['type'].lower().find('flux')>=0:
                            # flux is a general flux density, heatFlux: W/m2 is not a general flux name
                            g = self.translate_value(bc['value'])
                            integrals_N.append(g*Tq*ds(i))
                        elif bc['type'] == 'HTC':  # FIXME: HTC is not a general name or general type, only for thermal analysis
                            #Robin, how to get the boundary value,  T as the first, HTC as the second,  does not apply to nonlinear PDE
                            Ta = self.translate_value(bc['ambient'])
                            htc = self.translate_value(bc['value'])  # must be specified in Constant or Expressed in setup dict
                            integrals_N.append( htc*(Ta-T)*Tq*ds(i))
                        else:
                            print('temperature boundary type`{}` is not supported thus ignored'.format(bc['type']))
                        '''
                else:
                    print('boundary setup is done in scalar transport for incompressible flow'.format(bc['variable']))
        ## end of boundary setup
        return Dirichlet_bcs_up, F_bc

    def solve_form(self, F, up_, Dirichlet_bcs_up):
        # only for static case?
        if self.using_nonlinear_solver:
            return self.solve_nonlinear_problem(F, up_, Dirichlet_bcs_up, self.J)
        else:        # Solve Navier-Stokes problem with Picard method
            iter_ = 0
            max_iter = 50
            eps = 1.0
            tol = 1E-4
            under_relax_ratio = 0.7
            up_temp = Function(self.function_space)  # a temporal to save value in the Picard loop

            timer_solver = Timer("TimerSolveStatic")
            timer_solver.start()
            while (iter_ < max_iter and eps > tol):
                # solve the linear stokes flow to avoid up_s = 0

                up_temp.assign(up_)
                # other solving methods
                up_ = self.solve_linear_problem(F, up_, Dirichlet_bcs_up)
                #  AMG is not working with mixed function space
                
                #limiting result value, if the problem is highly nonlinear
                diff_up = up_.vector().array() - up_temp.vector().array()
                eps = np.linalg.norm(diff_up, ord=np.Inf)

                print("iter = {:d}; eps_up = {:e}; time elapsed = {}\n".format(iter_, eps, timer_solver.elapsed()))

                ## underreleax should be defined here, Courant number,
                up_.vector()[:] = up_temp.vector().array() + diff_up * under_relax_ratio

                iter_ += 1
            ## end of Picard loop
            timer_solver.stop()
            print("*" * 10 + " end of Navier-Stokes Picard iteration" + "*" * 10)

            return up_

    def plot_result(self):
        if self.solving_temperature:
            u,p,T= split(self.result)
            if self.using_matplotlib:
                import matplotlib.pyplot as plt
                plt.figure()
                plot(u ,title = "velocity")
                plt.figure()
                plot(p, title = "pressure", scalarbar = True)
                plt.gca().legend(loc = 'best')
                plt.figure()
                plot(T, title = "temperature", scalarbar = True)
                plt.gca().legend(loc = 'best')
            else:
                plot(u)
                plot(p)
                plot(T)
        else:
            u,p= split(self.result)
            plot(u)
            plot(p)
