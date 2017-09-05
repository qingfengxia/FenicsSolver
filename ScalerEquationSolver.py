from __future__ import print_function, division
import math
import numpy as np

from dolfin import *

supported_scaler_equations = {'temperature'}
# small species factor, that mixed material properties are same as the primary species, such as dye in water

#thermal diffusivity = (thermal conductivity) / (density * specific heat)
# thermal capacity = density * specific heat

from SolverBase import SolverBase, SolverError
class ScalerEquationSolver(SolverBase):
    """  this is a general scaler solver, modelled after EnergyEquation, other solver can derive from this basic one
    # 3 types of boundaries supported:
    # energy source unit:  W/m^3
    # convective velocity: m/s, 
    # Thermal Conductivity:  w/(K m)
    # Specific Heat Capacity, Cp:  J/(kg K)
    # shear_heating,  common lubrication scinario, high viscosity and shear speed, counted as volume source
    """
    def __init__(self, s):
        SolverBase.__init__(self, s)

        if 'scaler_name' in self.settings:
            self.scaler_name = self.settings['scaler_name'].lower()
        else:
            self.scaler_name = "temperature"

        if 'convective_velocity' in self.settings:
            self.convective_velocity = self.settings['convective_velocity']
        else:
            self.convective_velocity = None

    def capacity(self):
        if 'capacity' in self.material:
            return self.material['capacity']
        # if not found, calc it
        if self.scaler_name == "temperature":
            return self.material['density'] * self.material['specific_heat_capacity']
        elif self.scaler_name == "spicies":
            pass
        elif self.scaler_name == "electric_potential":
            pass
        else:
            raise SolverError(''.format(self.scaler_name))
            
    def conductivity(self):
        if 'conductivity' in self.material:
            return self.material['conductivity']
        # if not found, calc it
        if self.scaler_name == "temperature":
            return self.material['thermal_conductivity']
        raise SolverError(''.format(self.scaler_name))

    def get_internal_field(self):
        v0 = self.translate_value(self.initial_values[self.scaler_name])
        if isinstance(v0, (Constant, Expression)):
            v0 = interpolate(v0, self.function_space)
        return v0

    def update_boundary_conditions(self, time_iter_, T_0, T_prev):
        T = TrialFunction(self.function_space)
        Tq = TestFunction(self.function_space)

        ds= Measure("ds", subdomain_data=self.boundary_facets)  # if later marking updating in this ds?
        # no default boundary type applied? 
        bcs = []
        integrals_N = []  # heat flux
        integrals_R = []  # HTC
        for name, bc in self.boundary_conditions.items():
            i = bc['boundary_id']
            if bc['type'] =='Dirichlet':
                T_bc = self.get_boundary_value(bc, time_iter_)
                dbc = DirichletBC(self.function_space, T_bc, self.boundary_facets, i)
                bcs.append(dbc)
            elif bc['type'] == 'Neumann':  # as heat_flux_density , zero gradient supported? yes
                g = self.get_boundary_value(bc, time_iter_)
                integrals_N.append( g*Tq*ds(i))  # normal direction?
            elif bc['type'] == 'Robin':
                htc, Ta = bc['value']  # must be specified in Constant or Expressed in setup dict
                integrals_R.append( htc*(T - Ta)*Tq*ds(i))  # normal direction?
            else:
                raise SolverError('boundary type`{}` is not supported'.format(bc['type']))

        # Energy equation
        def F_static(T, Tq):
            k = self.conductivity() # constant, experssion or tensor
            F =  inner(k * grad(T), grad(Tq))*dx
            if self.convective_velocity:  # convective heat conduction
                self.vector_space = VectorFunctionSpace(self.function_space.mesh(), 'CG', self.degree)
                vel = project(self.convective_velocity, self.vector_space)
                F += inner(inner(k * grad(T), vel), Tq)*dx # support k as tensor for anisotropy conductivity
            F += sum(integrals_R) - sum(integrals_N)  # 
            if self.body_source:
                F -= self.body_source*Tq*dx
            return F

        if self.transient_settings['transient']:
            dt = self.get_time_step(time_iter_)
            theta = Constant(0.5) # Crank-Nicolson time scheme
            # Define time discretized equation, it depends on scaler type:  Energy, Species,
            capacity = self.capacity(time_iter_)
            F = capacity * (1.0/dt)*inner(T-T_prev, Tq)*dx \
                   + theta*F_static(T, Tq) + (1.0-theta)*F_static(T_0, Tq)
        else:
            F = F_static(T, Tq)
        print(F)
        return F, bcs

    def solve_static(self, F, T_0=None, bcs=[]):
        # solving
        a_T, L_T = system(F)
        A_T = assemble(a_T)
        solver_T= KrylovSolver('gmres', 'ilu')

        b_T = assemble(L_T)
        [bc.apply(A_T, b_T) for bc in bcs]  # apply Dirichlet BC

        T = Function(self.function_space)
        if T_0:
            T.vector()[:] = T_0.vector().array()
        solver_T.solve(A_T, T.vector(), b_T)

        return T

    ############## public API ##########################
    def solve(self):
        self.result = self.solve_transient()
        return self.result

    def export(self):
        #save and return save file name, also timestamp
        result_filename = self.settings['case_folder'] + os.path.sep + "temperature" + "_time0" +  ".vtk"

    def plot(self):
        plot(self.result)
