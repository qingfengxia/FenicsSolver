from __future__ import print_function, division
#import math may cause error
import numbers
import numpy as np
import os.path

from dolfin import *

class SolverError(Exception):
    pass

default_case_settings = {'solver_name': None,
                'case_name': 'test', 'case_folder': "/tmp/",  'case_file': None,
                'mesh':  None, 'function_space': None, 'periodic_boundary': None, 
                'boundary_conditions': None, 
                'body_source': None,
                'initial_values': {},
                'material':{},
                'solver_settings': {
                    'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.01, 'ending_time': 0.03},
                    'convergence_settings': {'default': 1e-3},
                    'reference_values': {},
                    }
                }

class SolverBase():
    """ shared base class for all fenics solver with utilty functions
    solve(), plot(), get_variables(), generate_
    """
    def __init__(self, case_input):
        if isinstance(case_input, (dict)):
            self.settings = case_input
            self.load_settings(case_input)
        else:
            raise SolverError('case setup data must be a python dict')

    def load_settings(self, s):
        #
        self.degree = 1
        if s['mesh'] and s['function_space'] ==None:
            if isinstance(s['mesh'], (str)):
                mesh = self.read_mesh(s['mesh'])
            else:
                mesh = s['mesh']
            self.generate_function_space(mesh, s['periodic_boundary'])
        elif s['mesh']==None and s['function_space']:
            self.function_space = s['function_space']
            self.mesh = self.function_space.mesh()
        else:
            raise SolverError('mesh or function space must specified to construct solver object')
        self.dimension = self.mesh.geometry().dim()

        ## boundary and body source
        self.boundary_conditions = s['boundary_conditions']
        #also read subdomains         #self.subdomains = MeshFunction("size_t", mesh, fname+"_physical_region.xml")
        if 'boundary_file' in s and os.path.exists(s['boundary_file']):
            self.boundary_facets = MeshFunction("size_t", self.mesh, s['boundary_file'])  # fname+"_facet_region.xml"
        else:
            self.generate_boundary_facets()
        if 'body_source' in s and s['body_source']:
            self.body_source = self.translate_value(s['body_source'])
        else:
            self.body_source = None

        ## initial and reference values
        self.initial_values = s['initial_values']
        self.reference_values = s['solver_settings']['reference_values']
        
        ## material
        self.material = s['material']
        
        ## solver setting, transient settings
        self.transient_settings = s['solver_settings']['transient_settings']
        """
        ts = s['solver_settings']['transient_settings']
        self.transient = ts['transient']
        self.starting_time = ts['starting_time']
        self.time_step = ts['time_step']
        self.ending_time = ts['ending_time']
        """

    def read_mesh(self, filename):
        if not os.path.exists(filename):
            raise SolverError('mesh file does not exist, check the file name')
        if filename[-5:] == ".xdmf":
            mesh = Mesh()
            f = XDMFFile(mpi_comm_world(), filename)
            f.read(mesh, True)
        elif filename[-4:] == ".xml":
            mesh = Mesh(filename)
        else:
            raise SolverError('mesh or function space must specified to construct solver object')
        return mesh

    def generate_function_space(self, mesh, periodic_boundary):
        self.degree = 1
        try:
            self.mesh = mesh
            if periodic_boundary:
                self.function_space = FunctionSpace(self.mesh, "CG", self.degree, constrained_domain=periodic_boundary)
                # the group and degree of the FE element.
            else:
                self.function_space = FunctionSpace(self.mesh, "CG", self.degree)
        except:
            raise SolverError('Fail to generate function space from mesh')
        self.is_mixed_function_space = False  # how to detect it is mixed?

    def generate_boundary_facets(self):
        boundary_facets = FacetFunction('size_t', self.mesh)
        boundary_facets.set_all(0)
        ## boundary conditions applying
        for name, bc in self.boundary_conditions.items():
            bc['boundary'].mark(boundary_facets, bc['boundary_id'])
        self.boundary_facets = boundary_facets
        #plot(boundary_facets, "boundary facets colored by ID")

    def translate_value(self, value):
        W = self.function_space
        if isinstance(value, tuple):
            if len(value) >= self.dimension:
                values_0 = Constant(value)
            else:
                values_0 = value  # fixme!!!
        elif isinstance(value, (Constant, Expression)):
            values_0 = value  # leave it as it is
        elif  isinstance(value, (numbers.Number)):
            values_0 = Constant(value)  # leave it as it is
        elif isinstance(value, (str, unicode)):
            if os.path.exists(value):
                #also possible continue from existent solution, or interpolate from diff mesh density
                values_0  << value
                import fenicstools
                values_0 = fenicstools.interpolate_nonmatching_mesh(values_0 , W)
            else:  # C++ expressing string
                values_0 = interpolate(Expression(value), W) 
        else:
            raise TypeError(' {} is supplied, not tuple, number, Constant,file name, Expression'.format(type(value)))
            #values_0 = None
        return values_0

    def get_boundary_value(self, bc, time_iter_=None):
        if self.transient_settings['transient'] and isinstance(bc['value'], list):  # if it is a list but not tuple
            bvalue = (bc['value'])[time_iter_] #already, interpolated function, should be
        else:
            bvalue = bc['value']
        if bc['value'] == 'Robin':  # value is a tuple of 2 element
            return bvalue
        else:
            return self.translate_value(bvalue)

    def get_time_step(self, time_iter_):
        ## fixed step, but could be supplied with an np.array/list
        try:
            dt = float(self.transient_settings['time_step'])
        except:
            if len(self.time_step) >= time_iter_:
                dt = self.time_step[time_iter_]
            else:
                print('time step can only be a sequence or scaler')
        self.mesh.hmin()  # Compute minimum cell diameter. courant number
        return dt

    def solve_transient(self):
        ## init to default or user provided constant, todo: possibly obtained from the previous time step
        up_0 = self.get_internal_field()

        # Define functions for transient loop
        up_prev = Function(self.function_space)
        ts = self.transient_settings
        # Define a parameters for a stationary loop
        t = ts['starting_time']
        time_iter_ = 0
        if ts['transient']:
            t_end = ts['ending_time']
        else:
            t_end = ts['time_step']

        # Transient loop also works for steady, by set `t_end = self.time_step`
        timer_solver_all = Timer("TimerSolveAll")
        timer_solver_all.start()
        while (t < t_end):
            dt = self.get_time_step(time_iter_)
            
            ## overloaded by derived classes
            F, Dirichlet_bcs_up = self.update_boundary_conditions(time_iter_, up_0, up_prev)

            up_prev.assign(up_0)  #
            up_0 = self.solve_static(F, up_0, Dirichlet_bcs_up)

            print("Current time = ", t, " TimerSolveAll = ", timer_solver_all.elapsed())
            # stop for steady case, or update time
            if not self.transient_settings['transient']:
                break
            time_iter_ += 1
            t += dt
        ## end of time loop
        timer_solver_all.stop()

        return up_0

    ####################################
    def solve_iteratively(self, F, Dirichlet_bcs, u):
        problem = LinearVariationalProblem(lhs(F), rhs(F), u, Dirichlet_bcs)
        solver = LinearVariationalSolver(problem)

        solver.parameters["linear_solver"] = 'default'
        solver.parameters["preconditioner"] = 'default'

        solver.solve()
        return u

    def solve_amg(self, F, bcs, u):
        A, b = assemble_system(lhs(F), rhs(F), bcs)
        # Create near null space basis (required for smoothed aggregation AMG).
        # The solution vector is passed so that it can be copied to generate compatible vectors for the nullspace.
        null_space = self.build_nullspace(self.function_space, u.vector())
        # Attach near nullspace to matrix
        as_backend_type(A).set_near_nullspace(null_space)

        # Create PETSC smoothed aggregation AMG preconditioner and attach near null space
        pc = PETScPreconditioner("petsc_amg")

        # Use Chebyshev smoothing for multigrid
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")

        # Improve estimate of eigenvalues for Chebyshev smoothing
        PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

        # Create CG Krylov solver and turn convergence monitoring on
        solver = PETScKrylovSolver("cg", pc)
        solver.parameters["monitor_convergence"] = True

        # Set matrix operator
        solver.set_operator(A)

        # Compute solution
        solver.solve(u.vector(), b)
        
        return u
    
    def build_nullspace(self, V, x):
        """Function to build null space for 2D and 3D elasticity"""

        # Create list of vectors for null space
        if self.dimension == 3:
            nullspace_basis = [x.copy() for i in range(6)]
            # Build translational null space basis
            V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
            V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
            V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

            # Build rotational null space basis
            V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
            V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
            V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
            V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
            V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
            V.sub(1).set_x(nullspace_basis[5], -1.0, 2);
        elif self.dimension == 2:
            nullspace_basis = [x.copy() for i in range(3)]
            V.sub(0).set_x(nullspace_basis[2], -1.0, 1);
            V.sub(1).set_x(nullspace_basis[2], 1.0, 0);
        else:
            raise Exception('only 2D or 3D is supported by nullspace')

        for x in nullspace_basis:
            x.apply("insert")

        # Create vector space basis and orthogonalize
        basis = VectorSpaceBasis(nullspace_basis)
        basis.orthonormalize()

        return basis