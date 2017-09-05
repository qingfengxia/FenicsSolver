from __future__ import print_function, division
import math
import numpy as np

from dolfin import *

supported_scaler_equations = {'temperature'}
# small species factor, that mixed material properties are same as the primary species, such as dye in water

# water
_default_material = {'conductivity': Constant(0.6),  # this is a general one, thermal, electrical, diff
    'specific_heat_capacity': Constant(4200), 
    'density' : Constant(1000),
    'kinetic_viscosity': Constant(1e06)
    }
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
    # shear_heating,  common in lubrication scinario, high viscosity and shear speed, one kind of volume/body source
    """
    def __init__(self, s):
        SolverBase.__init__(self, s)

        if 'scaler_name' in self.settings:
            self.scaler_name = self.settings['scaler_name'].lower()
        else:
            self.scaler_name = "temperature"

        if 'convective_velocity' in self.settings:
            self.convective_velocity = self.settings['convective_velocity']

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

    def solve(self):
        return self.solve_transient()

    def get_internal_field(self):
        v0 = self.translate_value(self.initial_values)
        if isinstance(v0, (Constant, Expression)):
            v0 = interpolate(v0, self.function_space)
        return v0

    def update_boundary_conditions(self, time_iter_, T_0, T_prev):
        # boundary type is defined in FreeCAD FemConstraintFluidBoundary and its TaskPanel
        # zeroGradient is default thermal boundary, no effect on equation
        T = TrialFunction(self.function_space)  # todo: could be shared beween time step
        Tq = TestFunction(self.function_space)
        normal = FacetNormal(self.mesh)

        ds= Measure("ds", subdomain_data=self.boundary_facets)  # normal direction
        bcs = []
        integrals_N = []  # heat flux
        k = self.conductivity() # constant, experssion or tensor
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
            elif bc['type'] == 'heatFlux': # heatFlux: W/m2, it is not a general flux name
                g = self.get_boundary_value(bc, time_iter_)
                integrals_N.append(g*Tq*ds(i))
            elif bc['type'] == 'mixed':
                T_bc, g = self.get_boundary_value(bc, time_iter_)
                integrals_N.append(k*g*Tq*ds(i))  # only work for constant conductivty k
                dbc = DirichletBC(self.function_space, T_bc, self.boundary_facets, i)
                bcs.append(dbc)
            elif bc['type'] == 'Robin' or bc['type'] == 'HTC':  # FIXME: HTC is not a general name
                #Robin, how to get the boundary value,  T as the first, HTC as the second
                Ta, htc = bc['value']  # must be specified in Constant or Expressed in setup dict
                integrals_N.append( htc*(Ta-T)*Tq*ds(i))
            else:
                raise SolverError('boundary type`{}` is not supported'.format(bc['type']))

        # Energy equation
        def F_static(T, Tq):
            F =  inner(k * grad(T), grad(Tq))*dx
            if self.convective_velocity:  # convective heat conduction
                self.vector_space = VectorFunctionSpace(self.function_space.mesh(), 'CG', self.degree)
                vel = project(self.convective_velocity, self.vector_space)
                F += inner(inner(k * grad(T), vel), Tq)*dx # support k as tensor for anisotropy conductivity
            F -= sum(integrals_N)
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
        return F, bcs

    def solve_static(self, F, T_0=None, bcs=[]):
        # solving
        a_T, L_T = system(F)
        A_T = assemble(a_T)

        # solver and parameters are defined by solver_parameters dict
        solver_T= KrylovSolver('gmres', 'ilu')  # not working with MPI
        solver_T.parameters["relative_tolerance"] = 1e-8
        solver_T.parameters["maximum_iterations"] = 5000
        #solver_T.parameters["monitor_convergence"] = True

        b_T = assemble(L_T)
        [bc.apply(A_T, b_T) for bc in bcs]  # apply Dirichlet BC

        T = Function(self.function_space)
        if T_0:
            T.vector()[:] = T_0.vector().array()
        solver_T.solve(A_T, T.vector(), b_T)

        return T
