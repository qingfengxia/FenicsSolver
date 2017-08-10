from __future__ import print_function, division
#import math may cause error
import numbers
import numpy as np
import os.path

from dolfin import *

#Oasis: a high-level/high-performance open source Navier-Stokes solve

if dolfin.MPI.size(dolfin.mpi_comm_world())>1:
    using_MPI = True
else:
    using_MPI = False
# Define a dolfin parameters
parameters["linear_algebra_backend"] = "PETSc"  #UMFPACK: out of memory, PETSc divergent
#parameters["linear_algebra_backend"] = "Eigen"  # 'uBLAS' is not supported any longer

parameters["mesh_partitioner"] = "SCOTCH"
#parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["optimize"] = True


"""
TODO:
1. ctor() using dict, load json case_input file
2. boundary translation
3. test parallel by MPI
"""

from SolverBase import SolverBase
class CoupledNavierStokesSolver(SolverBase):
    """  incompressible and laminar only
    """
    def __init__(self, case_input):

        SolverBase.__init__(self, case_input)

        #self.bcs_pressure = case_input['pressure_boundary_conditions']
        #self.bcs_velocity = case_input['pressure_boundary_conditions']

        if case_input['body_source']:  # FIXME: source term type, centrifugal force is possible
            self.body_force = self.translate_value(case_input['body_source'])
        else:
            if self.dimension == 2: 
                self.body_force = Constant((0, -9.8))
            else:
                self.body_force = Constant((0, 0, -9.8))

        ## default ref and init value
        self.reference_values = {'pressure': 1e5, 'velocity': 1, 'temperature': 293, 'length': 1 }
        if not self.initial_values:
            if self.dimension == 3:
                self.initial_values = {'velocity': (0,0,0), 'pressure': self.reference_values['pressure'] }
            else:
                self.initial_values = {'velocity': (0,0), 'pressure': self.reference_values['pressure'] }

        ## Define solver parameters, underreleax, tolerance, max_iter
        self.is_iterative_solver = True

        if using_MPI:
            self.solver_settings['linear_solver'] = 'bicgstab'  # "gmres" # not usable in MPI
            self.solver_settings['preconditioner']= "hypre_euclid"
        else:
            self.solver_settings['linear_solver'] = 'default'  # 
            self.solver_settings['preconditioner'] = "default"  # 'default', ilu only works in serial

    def generate_function_space(self, mesh, periodic_boundary):
        self.vel_degree = 2

        self.mesh = mesh
        V = VectorElement("CG", self.mesh.ufl_cell(), self.vel_degree)  # degree 2, must be higher than pressure
        Q = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        #temperature
        #T = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        #mixed_element = [V, Q]  # MixedFunctionSpace has been removed from 2016.2
        if periodic_boundary:
            self.function_space = FunctionSpace(self.mesh, V * Q, constrained_domain=periodic_boundary)  # new API
        else:
            self.function_space = FunctionSpace(self.mesh, V * Q)

        self.is_mixed_function_space = False  # how to detect it is mixed?

    def get_internal_field(self):
        up0 = Function(self.function_space)
        #u0, p0 = split(up0)
        #print(self.initial_values, self.function_space)
        #u0 = interpolate(self.translate_value(self.initial_values['velocity']), self.function_space.sub(0))
        #p0 = interpolate(self.translate_value(self.initial_values['pressure']), self.function_space.sub(1))
        return up0

    def viscous_stress(self, u, p):
        T_space = TensorFunctionSpace(self.function_space.mesh(), 'CG', self.vel_degree)
        sigma = project(self.viscosity()*(grad(u) + grad(u).T) - p*Identity(), T_space, 
            solver_type = 'mumps', form_compilder_parameters = {'cpp_optimize':  True, "representation": 'quadrature', "quadrature_degree": 2})
        return sigma

    def viscous_heat(self, u, p):
        # shear heating
        #V_space = VectorFunctionSpace(self.function_space.mesh())
        V_space = self.function_space.sub(0)
        return project(inner(self.viscosity()*(grad(u) + grad(u).T) - p*Identity(), u) , V_space,
            solver_type = 'mumps', form_compilder_parameters = {'cpp_optimize':  True, "representation": 'quadrature', "quadrature_degree": 2})

    def viscosity(self):
        #def viscosity_function(u, p):
        #    return Constant(1)*pow(p/Constant(self.reference_values['pressure']), 2)
        return self.material['kinetic_viscosity']  # nonlinear, nonNewtonian

    def update_boundary_conditions(self, time_iter_, up_0, up_prev):
        W = self.function_space
        boundary_facets = FacetFunction('size_t', W.mesh())
        boundary_facets.set_all(0)
        ## boundary setup and update for each time step
        n = FacetNormal(self.mesh)  # used in pressure force

        ## boundary conditions applying
        for key, bc in self.boundary_conditions.items():  # not checked
            bc['boundary'].mark(boundary_facets, bc['boundary_id'])
        ds = Measure("ds", subdomain_data=boundary_facets)
        if time_iter_ == 0:
            plot(boundary_facets, title ="boundary colored by ID")  # diff color do visual diff boundary 

        # Define unknown and test function(s)
        v, q = TestFunctions(W)
        up = TrialFunction(W)
        u, p = split(up)

        u_0, p_0 = split(up_0)

        ## weak form
        F = self.F_static(u, v, u_0, p, q, p_0)
        if self.transient:  # temporal scheme
            u_prev, p_prev = split(up_1)
            F += (1 / self.get_time_step(time_iter_)) * inner(u - u_prev, v) * dx

        Dirichlet_bcs_up = []
        #Dirichlet_bcs_up = [DirichletBC(W.sub(0), Constant(0,0,0), boundary_facets, 0)] #default vel
        for key, bc in self.boundary_conditions.items():
            if bc['variable'] == 'velocity':
                bvalue = self.get_boundary_value(bc, time_iter_)
                if bc['type'] == 'Dirichlet':
                    Dirichlet_bcs_up.append(DirichletBC(W.sub(0), bvalue, boundary_facets, bc['boundary_id']) )
                elif bc['type'] == 'Neumann':  # zero gradient, outflow
                    pass # FIXME
                else:
                    print('velocity boundary type`{}` is not supported'.format(bc['type']))

            #Dirichlet_bcs_up.append(DirichletBC(W.sub(1), Constant(self.pressure_ref), boundary_facets, 0))  # not correct
            if bc['variable'] == 'pressure':
                bvalue = self.get_boundary_value(bc, time_iter_)
                if bc['type'] == 'Dirichlet':  # pressure  inlet or outlet
                    Dirichlet_bcs_up.append(DirichletBC(W.sub(1), bvalue, boundary_facets, bc['boundary_id']) )
                    #  viscous force on pressure boundary?
                    F += inner(bvalue*n, v)*ds(bc['boundary_id'])  # very import to make sure convergence
                    F -= self.viscosity()*inner((grad(u) + grad(u).T)*n, v)*ds(bc['boundary_id'])  #  why 
                elif bc['type'] == 'Neumann':  # zero gradient
                    pass   # FIXME, not very common boundary
                else:
                    print('pressure boundary type`{}` is not supported thus ignored'.format(bc['type']))
        ## end of boundary setup
        return F, Dirichlet_bcs_up

    def F_static(self, u, v, u_0, p, q, p_0):
        def epsilon(u):
            """Return the symmetric gradient."""
            return 0.5 * (grad(u) + grad(u).T)

        _nu = self.viscosity()
        if isinstance(_nu, (Constant, numbers.Number)):
            nu = Constant(_nu)
        #else:  #if nu is nu is string or function
        #nu = viscosity(u, p)

        # Define Form for the static Stokes Coupled equation,
        F = nu * 2.0*inner(epsilon(u), epsilon(v))*dx \
            - p*div(v)*dx \
            + div(u)*q*dx
        if self.body_force: 
            F -= inner(self.body_force, v)*dx
        # Add convective term
        F += inner(dot(grad(u), u_0), v)*dx

        return F

    def solve_nonlinear(self, time_iter_, up_0, up_prev):
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

    def solve_static(self, F, up_0, Dirichlet_bcs_up):
        # Solve stationary Navier-Stokes problem with Picard method
        # other methods may be more acurate and faster

        #Picard loop
        up_s = Function(self.function_space)
        if up_0:     up_s.vector()[:] = up_0.vector().array()  # init
        u_s, p_s = split(up_s)

        iter_ = 0
        max_iter = 50
        eps = 1.0
        tol = 1E-3

        timer_solver = Timer("TimerSolveStatic")
        timer_solver.start()
        while (iter_ < max_iter and eps > tol):
            # solve the linear stokes flow to avoid up_s = 0
            #solve(F, up_s, Dirichlet_bcs_up)  #can solve nonline weak form

            problem = LinearVariationalProblem(lhs(F), rhs(F), up_s, Dirichlet_bcs_up)
            solver = LinearVariationalSolver(problem)

            solver.parameters["linear_solver"] = 'default'
            solver.parameters["preconditioner"] = 'default'

            solver.solve()
            
            #up_s = self.solve_iteratively(F, Dirichlet_bcs_up, up_s)
            #up_s = self.solve_amg(F, Dirichlet_bcs_up, up_s)  #  AMG is not working with mixed function space

            diff_up = up_s.vector().array() - up_0.vector().array()
            eps = np.linalg.norm(diff_up, ord=np.Inf)

            print("iter = {:d}; eps_up = {:e}; time elapsed = {}\n".format(iter_, eps, timer_solver.elapsed()))

            ## underreleax should be defined here, Courant number, 
            up_0.vector()[:] = up_0.vector().array() + diff_up * 0.7

            iter_ += 1
        ## end of Picard loop
        timer_solver.stop()
        print("*" * 10 + " end of Navier-Stokes equation iteration" + "*" * 10)

        return up_0

    def solve(self):
        self.result = self.solve_transient()
        return self.result

    def plot(self):
        pass
