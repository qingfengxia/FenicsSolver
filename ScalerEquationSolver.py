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
            raise SolverError('material capacity property is not found for {}'.format(self.scaler_name))

    def conductivity(self):
        if 'conductivity' in self.material:
            return self.material['conductivity']
        # if not found, calc it
        if self.scaler_name == "temperature":
            return self.material['thermal_conductivity']
        raise SolverError('conductivity material property is not found for {}'.format(self.scaler_name))

    def get_internal_field(self):
        v0 = self.translate_value(self.initial_values[self.scaler_name])
        if isinstance(v0, (Constant, Expression)):
            v0 = interpolate(v0, self.function_space)
        return v0

    def generate_form(self, time_iter_, T, Tq, T_0, T_prev):
        #T = TrialFunction(self.function_space)  # todo: could be shared beween time step
        #Tq = TestFunction(self.function_space)  # todo: could be shared beween time step
        normal = FacetNormal(self.mesh)

        ds= Measure("ds", subdomain_data=self.boundary_facets)  # normal direction
        bcs = []
        integrals_N = []  # heat flux
        k = self.conductivity() # constant, experssion or tensor

        # TODO: split into a new function update_boundary
        # boundary type is defined in FreeCAD FemConstraintFluidBoundary and its TaskPanel
        # zeroGradient is default thermal boundary, no effect on equation
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
        #print(F)
        return F, bcs

    def solve_static(self, F, T, bcs=[]):
        # solving
        a_T, L_T = system(F)
        A_T = assemble(a_T)

        parameters = self.solver_settings['solver_parameters']
        # solver and parameters are defined by solver_parameters dict
        if has_petsc():
            solver_T= PETScKrylovSolver('default', 'default')
        else:
            solver_T= KrylovSolver('default', 'default')  # 'gmres', 'ilu', not working with MPI
        solver_T.parameters["relative_tolerance"] = parameters["relative_tolerance"] 
        solver_T.parameters["maximum_iterations"] = parameters["maximum_iterations"]
        solver_T.parameters["monitor_convergence"] = parameters["monitor_convergence"]

        b_T = assemble(L_T)
        #for bc in bcs: print(type(bc))
        [bc.apply(A_T, b_T) for bc in bcs]  # apply Dirichlet BC

        solver_T.solve(A_T, T.vector(), b_T)

        return T

    ############## public API ##########################
    def solve(self):
        self.result = self.solve_transient()
        return self.result

    def export(self):
        #save and return save file name, also timestamp
        result_filename = self.settings['case_folder'] + os.path.sep + "temperature" + "_time0" +  ".vtk"
        return result_filename

    def plot(self):
        plot(self.result)
