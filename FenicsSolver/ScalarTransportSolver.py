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

supported_scalars = {'temperature', 'electric_potential', 'species_concentration'}
electric_permittivity_in_vacumm = 8.854187817e-12
# For small species factor, diffusivity = primary species, such as dye in water, always with convective velocity
# electric_potential, only for dieletric material electrostatics (permittivity/conductivity << 1)
# magnetic_potential is a vector, magnetostatics (static current) is solved in MaxwellEMSolver (permittivity/conductivity >> 1)
# porous pressure, e.g. underground water pressure 

# thermal diffusivity = (thermal conductivity) / (density * specific heat)
# thermal volumetric capacity = density * specific heat

from .SolverBase import SolverBase, SolverError
class ScalarTransportSolver(SolverBase):
    """  general scalar transportation (diffusion and advection) solver, exampled by Heat Transfer
    # 4 types of boundaries supported: math, physical 
    # body source unit:  W/m^3, apply to whole body/domain
    # surface source unit: W/m^2, apply to whole boundary,  
    # convective velocity: m/s, stablization is controlled by advection_settings
    # Thermal Conductivity:  w/(K m)
    # Specific Heat Capacity, Cp:  J/(kg K)
    # thermal specific:
    # shear_heating: common in lubrication scinario, high viscosity and high shear speed, one kind of volume/body source
    # radiation:  radiation_settings {}
    """
    def __init__(self, s):
        SolverBase.__init__(self, s)

        if 'scalar_name' in self.settings:
            self.scalar_name = self.settings['scalar_name'].lower()
        else:
            self.scalar_name = "temperature"
        self.using_diffusion_form = False  # diffusion form is simple in math, but not easy to deal with nonlinear material property

        self.nonlinear = False
        self.nonlinear_material = True
        for v in self.material.values():
            if callable(v):  # fixedme: if other material properties are functions, it will be regarded as nonlinear
                self.nonlinear = True

        if self.scalar_name == "eletric_potential":
            assert self.settings['transient_settings']['transient'] == False
        #delay the convective velocity and radiation setting detection in geneate_form()

    def capacity(self, T=None):
        # to calc diffusion coeff : conductivity/capacity, it must be number only for 
        if 'capacity' in self.material:
            c = self.material['capacity']
        # if not found, calc it, otherwise, it is 
        elif self.scalar_name == "temperature":
            cp = self.material['specific_heat_capacity']
            c = self.material['density'] * cp
        elif self.scalar_name == "electric_potential":
            c = electric_permittivity_in_vacumm
        elif self.scalar_name == "spicies_concentration":
            c = 1
        else:
            raise SolverError('material capacity property is not found for {}'.format(self.scalar_name))
        #print(type(c))
        from inspect import isfunction
        if isfunction(c):  # accept only function or lambda,  ulf.algebra.Product is also callable
            self.nonlinear_material = True
            return c(T)
        return self.get_material_value(c)  # todo: deal with nonlinear material

    def diffusivity(self, T=None):
        if 'diffusivity' in self.material:
            c = self.material['diffusivity']
        elif self.scalar_name == "temperature":
            c = self.material['thermal_conductivity'] / self.capacity()
        elif self.scalar_name == "electric_potential":
            c = self.material['relative_electric_permittivity']
        elif self.scalar_name == "spicies_concentration":
            c = self.material['diffusivity']
        else:
            raise SolverError('conductivity material property is not found for {}'.format(self.scalar_name))

        from inspect import isfunction  # dolfin.Funciton is also callable
        if isfunction(c):
            self.nonlinear_material = True
            return c(T)
        return self.get_material_value(c)  # todo: deal with nonlinear material

    def conductivity(self, T=None):
        # nonlinear material:  c = function(T)
        if 'conductivity' in self.material:
            c = self.material['conductivity']
        elif self.scalar_name == "temperature":
            c = self.material['thermal_conductivity']
        elif self.scalar_name == "electric_potential":
            c = self.material['relative_electric_permittivity'] * electric_permittivity_in_vacumm
        elif self.scalar_name == "spicies_concentration":
            c = self.material['diffusivity']
        else:
            c = self.diffusivity() * self.capacity()
        #print('conductivity', c)
        from inspect import isfunction
        if isfunction(c): 
            self.nonlinear_material = True
            return c(T)
        return self.get_material_value(c)  # todo: deal with nonlinear material

    def get_convective_velocity_function(self, convective_velocity):
        import ufl.tensors  # used in coupled NS and energy form
        if isinstance(convective_velocity, ufl.tensors.ListTensor):
            return convective_velocity
        else:
            self.vector_function_space = VectorFunctionSpace(self.mesh, 'CG', self.settings['fe_degree']+1)
            vel = self.translate_value(convective_velocity, self.vector_function_space)
            #print('type of convective_velocity', type(convective_velocity), type(vel))
            #print("vel.ufl_shape", vel.ufl_shape)
            return vel

    def update_boundary_conditions(self, time_iter_, T, Tq, ds):
        # test_function is removed from integrals_N items, so SPUG residual can include boundary condition
        capacity = self.capacity(T) # constant, experssion or tensor

        bcs = []
        integrals_N = []
        if 'point_source' in self.settings and self.settings['point_source']:
            ps = self.settings['point_source']
            # FIXME: not test yet, assuming PointSource type, or a list of PointSource(value, position)
            bcs.append(ps)

        mesh_normal = FacetNormal(self.mesh)  # n is predefined as outward as positive
        if 'surface_source' in self.settings and self.settings['surface_source']:
            gS = self.get_flux(self.settings['surface_source']['value'])
            if 'direction' in self.settings['surface_source'] and self.settings['surface_source']['direction']:
                direction_vector = self.settings['surface_source']['direction']
            else:
                integrals_N.append(dot(mesh_normal*gS, v)*ds)

        for name, bc_settings in self.boundary_conditions.items():
            i = bc_settings['boundary_id']
            bc = self.get_boundary_variable(bc_settings)  # should deal with 'value' and 'values = []'

            if bc['type'] == 'Dirichlet' or bc['type'] == 'fixedValue':
                if not isinstance(bc['value'], DirichletBC):
                    T_bc = self.translate_value(bc['value'])
                    dbc = DirichletBC(self.function_space, T_bc, self.boundary_facets, i)
                    bcs.append(dbc)
                else:
                    bcs.append(bc['value'])
            elif bc['type'] == 'Neumann' or bc['type'] =='fixedGradient':  # unit: K/m
                g = self.translate_value(bc['value'])
                if self.using_diffusion_form:
                    integrals_N.append(g*Tq*ds(i))
                else:
                    integrals_N.append(capacity*g*Tq*ds(i))
                #integrals_N.append(inner(capacity * (normal*g), Tq)*ds(i))  # not working
            elif bc['type'] == 'symmetry':
                pass  # zero gradient
            elif bc['type'] == 'mixed' or bc['type'] == 'Robin':
                T_bc = self.translate_value(bc['value'])
                g = self.translate_value(bc['gradient'])
                if self.using_diffusion_form:  # solve T
                    integrals_N.append(g*Tq*ds(i))
                else:  # solver flux
                    integrals_N.append(capacity*g*Tq*ds(i))
                dbc = DirichletBC(self.function_space, T_bc, self.boundary_facets, i)
                bcs.append(dbc)
            elif bc['type'].lower().find('flux')>=0 or bc['type'] == 'electric_current':
                # flux is a general flux density, heatFlux: W/m2 is not a general flux name
                g = self.translate_value(bc['value'])
                if self.using_diffusion_form:
                    integrals_N.append(g/capacity*Tq*ds(i))
                else:
                    integrals_N.append(g*Tq*ds(i))
            elif bc['type'] == 'HTC':  # FIXME: HTC is not a general name or general type, only for thermal analysis
                #Robin, how to get the boundary value,  T as the first, HTC as the second,  does not apply to nonlinear PDE
                Ta = self.translate_value(bc['ambient'])
                htc = self.translate_value(bc['value'])  # must be specified in Constant or Expressed in setup dict
                if self.using_diffusion_form:
                    integrals_N.append( htc/capacity*(Ta-T)*Tq*ds(i))
                else:
                    integrals_N.append( htc*(Ta-T)*Tq*ds(i))
            else:
                raise SolverError('boundary type`{}` is not supported'.format(bc['type']))
        return bcs, integrals_N

    def get_body_source_items(self, time_iter_, T, Tq, dx):
        bs = self.get_body_source()  # defined in base solver, has already translated value
        print("body source: ", bs)
        if bs and isinstance(bs, dict):
            S = []
            for k,v in bs.items():
                # it is good to using DG for multi-scale meshing, subdomain marking double
                S.append(v['value']*Tq*dx(v['subdomain_id']))
            return S
        else:
            if bs:
                return [bs*Tq*dx]
            else:
                return None

    def generate_form(self, time_iter_, T, T_test, T_current, T_prev):
        # T, Tq can be shared between time steps, form is unified diffussion coefficient
        normal = FacetNormal(self.mesh)

        dx= Measure("dx", subdomain_data=self.subdomains)  # cells
        ds= Measure("ds", subdomain_data=self.boundary_facets)  #boundary cells
        #dS = Measure("dS", subdomain_data=self.boundary_facets)  

        conductivity = self.conductivity(T) # constant, experssion or tensor, function of T for nonlinear
        print('conductivity = ', conductivity)
        capacity = self.capacity(T)  # density * specific capacity -> volumetrical capacity
        print('capacity = ', capacity)
        #diffusivity = self.diffusivity(T)  # diffusivity not in used for this conductivity form
        #print("diffusivity = ", diffusivity)

        # detection here, to deal with assigned velocity after solver intialization
        if not hasattr(self, 'convective_velocity'):  # if velocity is not directly assigned to the solver
            if 'convective_velocity' in self.settings and self.settings['convective_velocity']:
                self.convective_velocity = self.settings['convective_velocity']
            else:
                self.convective_velocity = None

        if self.convective_velocity:
            if 'advection_settings' in self.settings:
                ads = self.settings['advection_settings']  # a very big panelty factor can stabalize, but leading to diffusion error
            else:
                ads = {'stabilization_method': None}  # default none

            velocity = self.get_convective_velocity_function(self.convective_velocity)
            h = 2*Circumradius(self.mesh)  # cell size

            if ads['stabilization_method'] == 'SPUG':
                # Add SUPG stabilisation terms
                print('solving convection by SPUG stablization')
                #`Numerical simulations of advection-dominated scalar mixing with applications to spinal CSF flow and drug transport` page 20
                # SPUG_method == 2, ref: 
                vnorm = sqrt(dot(velocity, velocity))
                Pe = ads['Pe']  # Peclet number: 
                tau = 0.5*h*pow(4.0/(Pe*h)+2.0*vnorm,-1.0)  # this user-chosen value
                delta = h/(2*vnorm)
                SPUG_method = 2
                if SPUG_method == 2:
                    Tq = (T_test + tau*inner(velocity, grad(T_test)))  # residual and variatonal form has diff sign for  the diffusion item
                elif SPUG_method == 1:
                    Tq = T_test
                else:
                    raise SolverError('SPUG only has only 2 variants')
            else:
                Tq = T_test
        else:
            Tq = T_test

        # poission equation, unified for all kind of variables
        # it apply to nonlinear conductivity, which is a function of temperature
        # d(conductivity)/ dT is ignored here
        def F_static(T, Tq):
            return  inner(conductivity * grad(T), grad(Tq))*dx

        if self.transient_settings['transient']:
            dt = self.get_time_step(time_iter_)
            theta = Constant(0.5) # Crank-Nicolson time scheme
            # Define time discretized equation, it depends on scalar type:  Energy, Species,
            # FIXME: nonlinear capacity is not supported
            F = (1.0/dt)*inner(T-T_prev, Tq)*capacity*dx \
                   + theta*F_static(T, Tq) + (1.0-theta)*F_static(T_prev, Tq)  # FIXME:  check using T_0 or T_prev ?
        else:
            F = F_static(T, Tq)

        bcs, integrals_N = self.update_boundary_conditions(time_iter_, T, Tq, ds)
        if integrals_N:
            F -= sum(integrals_N)

        bs_items = self.get_body_source_items(time_iter_,T, Tq, dx)
        if bs_items:
            F -= sum(bs_items)

        if self.convective_velocity:
            if self.nonlinear_material:
                F += inner(velocity, grad(T*capacity))*Tq*dx  # those 2 are equal
                #F += inner(velocity, grad(T))*Tq*capacity*dx + inner(velocity, grad(T))*Tq*self.material['dc_dT']*T*dx
            else:
                F += inner(velocity, grad(T))*Tq*capacity*dx
            if ads['stabilization_method'] and ads['stabilization_method'] == 'IP':
                print('solving convection by interior penalty stablization')
                alpha = Constant(ads['alpha'])
                F +=  alpha*avg(h)**2*inner(jump(grad(T),normal), jump(grad(Tq),normal))*capacity*dS
                # http://www.karlin.mff.cuni.cz/~hron/fenics-tutorial/convection_diffusion/doc.html
            if ads['stabilization_method'] and ads['stabilization_method'] == 'SPUG' and SPUG_method == 1:
                #https://fenicsproject.org/qa/6951/help-on-supg-method-in-the-advection-diffusion-demo/
                if self.transient_settings['transient']:
                    residual = dot(velocity, grad(T)) - theta*conductivity*div(grad(T)) - (1.0-theta)*conductivity*div(grad(T_prev)) \
                                    + (1.0/dt)*inner(T-T_prev, Tq)*capacity # FIXME:
                else:
                    residual = dot(velocity, grad(T)) - conductivity*div(grad(T))  # diffusion item sign is different from variational form
                F_residual = residual * delta*dot(velocity, grad(Tq)) * dx
                bs_r_items = self.get_body_source_items(time_iter_,T, delta*dot(velocity, grad(Tq)), dx)
                if bs_r_items:
                    F_residual -= sum(bs_r_items)
                F += F_residual

        using_mass_conservation = False # not well tested, Nitsche boundary
        if using_mass_conservation:
            print('mass conservation compensation for zero mass flux on the curved boundary')
            sigma = Constant(2) # penalty parameter
            #he = self.mesh.ufl_cell().max_facet_edge_length,    T - Constant(300)
            #F -= inner(dot(velocity, normal), dot(grad(T), normal))*Tq*capacity*ds  # (1.0/ h**sigma) *
            F -= dot(dot(velocity, normal), T)*capacity*ds*Tq

        if self.scalar_name == "temperature":
            if ('radiation_settings' in self.settings and self.settings['radiation_settings']):
                self.radiation_settings = self.settings['radiation_settings']
                self.has_radiation = True
            elif hasattr(self, 'radiation_settings') and self.radiation_settings:
                self.has_radiation = True
            else:
                self.has_radiation = False

            if self.has_radiation:
                #print(m_, radiation_flux, F)
                self.nonlinear = True
                F -= self.radiation_flux(T)*Tq*ds # for all surface, without considering view angle
        
        #print(F)
        if self.nonlinear_material:
            self.nonlinear_material = True
        if self.nonlinear:
            F = action(F, T_current)  # API 1.0 still working ; newer API , replacing TrialFunction with Function for nonlinear 
            self.J = derivative(F, T_current, T)  # Gateaux derivative

        return F, bcs

    def radiation_flux(self, T):
            Stefan_constant = 5.670367e-8  # W/m-2/K-4
            if 'emissivity' in self.material:
                emissivity = self.material['emissivity']  # self.settings['radiation_settings']['emissivity'] 
            elif 'emissivity' in self.radiation_settings:
                emissivity = self.radiation_settings['emissivity'] 
            else:
                emissivity = 1.0
            if 'ambient_temperature' in self.radiation_settings:
                T_ambient_radiaton = self.radiation_settings['ambient_temperature']
            else:
                T_ambient_radiaton = self.reference_values['temperature']

            m_ = emissivity * Stefan_constant
            radiation_flux = m_*(T_ambient_radiaton**4 - pow(T, 4))  # it is nonlinear item
            return radiation_flux

    def solve_form(self, F, T_current, bcs):
        if self.nonlinear:
            print('solving by nonlinear solver')
            return self.solve_nonlinear_problem(F, T_current, bcs, self.J)
        else:
            return self.solve_linear_problem(F, T_current, bcs)

    ############## public API ##########################

    def export(self):
        #save and return save file name, also timestamp
        result_filename = self.settings['case_folder'] + os.path.sep + self.get_variable_name() + "_time0" +  ".vtk"
        return result_filename

