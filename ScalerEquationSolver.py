from __future__ import print_function, division
import math
import numpy as np

from dolfin import *

supported_scaler_equations = {'temperature', 'electric_potential', 'species_concentration'}
# For small species factor, diffusivity = primary species, such as dye in water, always with convective velocity
# electric_potential -> difference between conductor and dieletric material
# magnetic_potential is a vector
# porous pressure, e.g. underground water pressure 

#thermal diffusivity = (thermal conductivity) / (density * specific heat)
# thermal capacity = density * specific heat

from SolverBase import SolverBase, SolverError
class ScalerEquationSolver(SolverBase):
    """  this is a general scaler solver, modelled after Heat Transfer Equation, other solver can derive from this basic one
    # 4 types of boundaries supported:
    # body source unit:  W/m^3
    # convective velocity: m/s, 
    # Thermal Conductivity:  w/(K m)
    # Specific Heat Capacity, Cp:  J/(kg K)
    # thermal specific:
    # shear_heating: common in lubrication scinario, high viscosity and high shear speed, one kind of volume/body source
    # radiation: 
    """
    def __init__(self, s):
        SolverBase.__init__(self, s)

        if 'scaler_name' in self.settings:
            self.scaler_name = self.settings['scaler_name'].lower()
        else:
            self.scaler_name = "temperature"

        if self.scaler_name == "eletric_potential":
            assert self.settings['transient_settings']['transient'] == False

        if self.scaler_name == "temperature":
            if 'radiation_settings' in self.settings:
                self.radiation_settings = self.settings['radiation_settings']
                self.has_radiation = True
            else:
                self.has_radiation = False

        if 'convective_velocity' in self.settings and self.settings['convective_velocity']:
            self.has_convection = True
            self.convective_velocity = self.settings['convective_velocity']
        else:
            self.has_convection = False
            self.convective_velocity = None

    def capacity(self):
        # to calc diffusion coeff : conductivity/capacity
        if 'capacity' in self.material:
            return self.material['capacity']
        # if not found, calc it
        if self.scaler_name == "temperature":
            return self.material['density'] * self.material['specific_heat_capacity']
        elif self.scaler_name == "electric_potential":
            return 1  # depends on source and boundary flux physical value, could be electric_permittivity
        elif self.scaler_name == "spicies_concentration":
            return 1
        else:
            raise SolverError('material capacity property is not found for {}'.format(self.scaler_name))

    def conductivity(self):
        if 'conductivity' in self.material:
            return self.material['conductivity']

        if self.scaler_name == "temperature":
            return self.material['thermal_conductivity']
        elif self.scaler_name == "electric_potential":
            return self.material['electric_permittivity']
        elif self.scaler_name == "spicies_concentration":
            return self.material['diffusivity']
        else:
            raise SolverError('conductivity material property is not found for {}'.format(self.scaler_name))

    def get_internal_field(self):
        #print(self.scaler_name, self.initial_values)
        v0 = self.translate_value(self.initial_values[self.scaler_name])
        if isinstance(v0, (Constant, Expression)):
            v0 = interpolate(v0, self.function_space)
        return v0

    def get_convective_velocity_function(self, convective_velocity):
        self.vector_space = VectorFunctionSpace(self.mesh, 'CG', self.degree+1)
        vel = self.translate_value(convective_velocity, self.vector_space)
        #vel = interpolate(Expression(('x[0]', 'x[1]'), degree = 1), self.vector_space)
        #vel = Constant((1.0, 1.0))
        #print("vel.ufl_shape", vel.ufl_shape)
        return vel

    def update_boundary_conditions(self, time_iter_, T, Tq, ds):
        k = self.conductivity() # constant, experssion or tensor
        bcs = []
        integrals_N = []  # heat flux
        for name, bc in self.boundary_conditions.items():
            i = bc['boundary_id']
            if bc['type'] == 'Dirichlet' or bc['type'] == 'fixedValue':
                T_bc = self.get_boundary_value(bc, time_iter_)
                dbc = DirichletBC(self.function_space, T_bc, self.boundary_facets, i)
                bcs.append(dbc)
            elif bc['type'] == 'Neumann' or bc['type'] =='fixedGradient':  # unit: K/m
                g = self.get_boundary_value(bc, time_iter_)
                integrals_N.append(k*g*Tq*ds(i))  # only work for constant conductivty k
                #integrals_N.append(inner(k * (normal*g), Tq)*ds(i))  # not working
            elif bc['type'] == 'flux' or bc['type'] == 'heatFlux' or  bc['type'] == 'electric_current':
                # flux is a general flux, heatFlux: W/m2, it is not a general flux name
                g = self.get_boundary_value(bc, time_iter_)
                integrals_N.append(g*Tq*ds(i))
            elif bc['type'] == 'mixed' or bc['type'] == 'Robin':
                T_bc, g = bc['value'], bc['gradient']
                integrals_N.append(k*g*Tq*ds(i))  # only work for constant conductivty k
                dbc = DirichletBC(self.function_space, T_bc, self.boundary_facets, i)
                bcs.append(dbc)
            elif bc['type'] == 'HTC':  # FIXME: HTC is not a general name or general type, only for thermal analysis
                #Robin, how to get the boundary value,  T as the first, HTC as the second
                Ta, htc = bc['ambient'], bc['value']  # must be specified in Constant or Expressed in setup dict
                integrals_N.append( htc*(Ta-T)*Tq*ds(i))
            else:
                raise SolverError('boundary type`{}` is not supported'.format(bc['type']))
        return bcs, integrals_N

    def generate_form(self, time_iter_, T, Tq, T_current, T_prev):
        # T, Tq can be shared between time steps, form is unified diffussion coefficient
        normal = FacetNormal(self.mesh)

        dx= Measure("dx", subdomain_data=self.subdomains)  # 
        ds= Measure("ds", subdomain_data=self.boundary_facets)

        k = self.conductivity() # constant, experssion or tensor
        capacity = self.capacity()  # density * specific capacity -> volumetrical capacity
        diffusivity = k / capacity  # diffusivity

        bcs, integrals_N = self.update_boundary_conditions(time_iter_, T, Tq, ds)
        # boundary type is defined in FreeCAD FemConstraintFluidBoundary and its TaskPanel
        # zeroGradient is default thermal boundary, no effect on equation?

        def get_source_item():
            if isinstance(self.body_source, dict):
                S = []
                for k,v in self.get_body_source().items():
                    # it is good to using DG for multi-scale meshing, subdomain marking double
                    S.append(v['value']*Tq*dx(v['subdomain_id']))
                return sum(S)
            else:
                if self.body_source:
                    return  self.get_body_source()*Tq*dx
                else:
                    return None

        # poission equation, unified for all kind of variables
        def F_static(T, Tq):
            F =  inner( diffusivity * grad(T), grad(Tq))*dx
            F -= sum(integrals_N) * Constant(1.0/capacity)
            return F

        def F_convective():
            h = CellSize(self.mesh)  # cell size
            velocity = self.get_convective_velocity_function(self.convective_velocity)
            if self.transient_settings['transient']:
                dt = self.get_time_step(time_iter_)
                # Mid-point solution
                T_mid = 0.5*(T_prev + T)
                # Residual
                res = (T - T_prev)/dt + (dot(velocity, grad(T_mid)) -  diffusivity * div(grad(T_mid)))  # does not support conductivity tensor
                # Galerkin variational problem
                F = Tq*(T - T_prev)/dt*dx + (Tq*dot(velocity, grad(T_mid))*dx +  diffusivity * dot(grad(Tq), grad(T_mid))*dx)
            else:
                T_mid = T
                # Residual
                res = dot(velocity, grad(T_mid)) -  diffusivity * div(grad(T_mid))
                #print(res)
                # Galerkin variational problem
                F = Tq*dot(velocity, grad(T_mid))*dx + \
                     diffusivity * dot(grad(Tq), grad(T_mid))*dx

            F -= sum(integrals_N) * Constant(1.0/capacity)  # included in F_static()
            if self.body_source:
                res -= get_source_item() * Constant(1.0/capacity)
                #F -= self.body_source*Tq*dx * capacity  # why F does not substract body?
            # Add SUPG stabilisation terms
            vnorm = sqrt(dot(velocity, velocity))
            F += (h/(2.0*vnorm))*dot(velocity, grad(Tq))*res*dx
            return F

        if self.convective_velocity:  # convective heat conduction
            F = F_convective()
        else:
            if self.transient_settings['transient']:
                dt = self.get_time_step(time_iter_)
                theta = Constant(0.5) # Crank-Nicolson time scheme
                # Define time discretized equation, it depends on scaler type:  Energy, Species,
                F = (1.0/dt)*inner(T-T_prev, Tq)*dx \
                       + theta*F_static(T, Tq) + (1.0-theta)*F_static(T_prev, Tq)  # FIXME:  check using T_0 or T_prev ? 
            else:
                F = F_static(T, Tq)
            #print(F, get_source_item())
            if self.body_source:
                F -= get_source_item() * Constant(1.0/capacity)

        if self.scaler_name == "temperature" and self.has_radiation:
            Stefan_constant = 5.670367e-8  # W/m-2/K-4
            if 'emissivity' in self.material:
                emissivity = self.material['emissivity']  # self.settings['radiation_settings']['emissivity'] 
            else:
                emissivity = 1.0
            T_ambient_radiaton = self.radiation_settings['ambient_temperature']
            m_ = emissivity * Stefan_constant
            radiation_flux = m_*(pow(T, 4) - T_ambient_radiaton**4)  # it is nonlinear item
            print(m_, radiation_flux, F)
            F -= radiation_flux*Tq*ds * Constant(1.0/capacity)  # for all surface, without considering view angle
            F = action(F, T_current)  # API 1.0 still working ; newer API , replacing TrialFunction with Function for nonlinear 
            self.J = derivative(F, T_current, T)  # Gateaux derivative
        return F, bcs

    def solve_static(self, F, T_current, bcs):
        if self.scaler_name == "temperature" and self.has_radiation:
            return self.solve_nonlinear_problem(F, T_current, bcs, self.J)
            """
            Stefan_constant = 5.670367e-8  # W/m-2/K-4
            emissivity = self.material['emissivity']
            T_ambient_radiaton = self.radiation_settings['ambient_temperature']
            m_ = emissivity * Stefan_constant
            #radiation_flux = interpolate(Expression("m_*(T - T_a)**4", m_ = m_, T= T, T_a = T_ambient_radiaton, \
            #        degree = self.degree), self.function_space)

            max_nonlinear_iteratons = 10
            T_ = Function(self.function_space)
            T_.assign(T_current)
            for i in range(max_nonlinear_iteratons):
                radiation_flux = m_*(pow(T_, 4) - T_ambient_radiaton**4)  # it is nonlinear item
                F_ = F - radiation_flux*Tq*ds   # for all surface, without considering view angle
                T_ = self.solve_linear_problem(F_, T_, bcs)
            return T_
            """
        else:
            return self.solve_linear_problem(F, T_current, bcs)

    ############## public API ##########################

    def export(self):
        #save and return save file name, also timestamp
        result_filename = self.settings['case_folder'] + os.path.sep + "temperature" + "_time0" +  ".vtk"
        return result_filename

