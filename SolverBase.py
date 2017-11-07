# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2017 - Qingfeng Xia <qingfeng.xia eng ox ac uk>                 *       *
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
import copy
import numpy as np
import os.path

from dolfin import *

class SolverError(Exception):
    pass

default_case_settings = {'solver_name': None,
                'case_name': 'test', 'case_folder': "/tmp/",  'case_file': None,
                'mesh':  None, 'function_space': None, 'periodic_boundary': None, 
                'boundary_conditions': None, 
                'body_source': None,  # dict for different subdomains {"sub_name": {'subdomain_id': 1, 'value': 2}}
                'initial_values': {},
                'material':{},  # can be a list of material dict for different subdomains
                'solver_settings': {
                    'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.01, 'ending_time': 0.03},
                    'reference_values': {},
                    'solver_parameters': {"relative_tolerance": 1e-5,  # mapping to solver.parameters of Fenics
                                                        "maximum_iterations": 500,
                                                        "monitor_convergence": False,  # print to console
                                                    },
                    },
                "output_settings": {}
                }

class SolverBase():
    """ shared base class for all fenics solver with utilty functions
    solve(), plot(), get_variables(), generate_
    """
    def __init__(self, case_input):
        if isinstance(case_input, (dict)):
            self.settings = case_input
            #self.print()
            self.load_settings(case_input)
        else:
            raise SolverError('case setup data must be a python dict')

    def print(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.settings)

    def load_settings(self, s):
        ## mesh and boundary
        self.boundary_conditions = s['boundary_conditions']
        if ('mesh' in s) and s['mesh']:
            if isinstance(s['mesh'], (str, unicode)):
                self.read_mesh(s['mesh'])  # it also read boundary
            elif isinstance(s['mesh'], (Mesh,)):
                self.mesh = s['mesh']
                self.generate_boundary_facets()
            else:
                raise SolverError('Error: mesh must be file path or Mesh object: {}')
            if 'periodic_boundary' not in s:  # check: settings file can not store None element?
                s['periodic_boundary'] = None
            self.generate_function_space(s['periodic_boundary'])
        elif ('mesh' not in s or s['mesh']==None) and ('function_space' in s and s['function_space']):
            self.function_space = s['function_space']
            self.degree = self.function_space._ufl_element.degree()
            self.mesh = self.function_space.mesh()
            self.generate_boundary_facets()
        else:
            raise SolverError('mesh or function space must specified to construct solver object')
        self.dimension = self.mesh.geometry().dim()

        ## 
        if 'body_source' in s and s['body_source']:
            self.body_source = s['body_source']
        else:
            self.body_source = None

        ## initial and reference values
        self.initial_values = s['initial_values']
        self.reference_values = s['solver_settings']['reference_values']
        
        ## material
        self.material = s['material']
        
        ## solver setting, transient settings
        self.solver_settings = s['solver_settings']
        self.transient_settings = s['solver_settings']['transient_settings']
        self.transient = self.transient_settings['transient']


    def _read_hdf5_mesh(self, filename):
        # path is identical to FenicsSolver.utility 
        mesh = Mesh()
        hdf = HDF5File(mesh.mpi_comm(), filename, "r")
        hdf.read(mesh, "/mesh", False)
        self.mesh = mesh

        self.subdomains = CellFunction("size_t", mesh)
        if (hdf.has_dataset("/subdomains")):
            hdf.read(subdomains, "/subdomains")
        else:
            print('Subdomain file is not provided')

        if (hdf.has_dataset("/boundaries")):
            self.boundary_facets = FacetFunction("size_t", mesh)
            hdf.read(self.boundary_facets, "/boundaries")
        else:
            print('Boundary facets file is not provided, marked from boundary settings')
            self.generate_boundary_facets()  # boundary marking from subdomain instance

    def _read_xml_mesh(self, filename):
        mesh = Mesh(filename)
        bmeshfile = filename[:-4] + "_facet_region.xml"
        self.mesh = mesh

        if os.path.exists(bmeshfile):
            self.boundary_facets = MeshFunction("size_t", mesh, bmeshfile)
        else:
            print('Boundary facets are not provided by xml input file, boundary will be marked from subdomain instance')
            self.generate_boundary_facets()  # boundary marking from subdomain instance

        subdomain_meshfile = filename[:-4] + "_physical_region.xml"
        if os.path.exists(subdomain_meshfile):
            self.subdomains = MeshFunction("size_t", mesh, subdomain_meshfile)
        else:
            self.subdomains = CellFunction("size_t", mesh)

    def read_mesh(self, filename):
        print(filename, type(filename))  # unicode filename NOT working, why?
        if isinstance(filename, (unicode,)):
            filename = filename.encode('utf-8')
        if not os.path.exists(filename):
            raise SolverError('mesh file: {} , does not exist'. format(filename))
        if filename[-5:] == ".xdmf":  # there are some new feature in 2017.2
            mesh = Mesh()
            f = XDMFFile(mpi_comm_world(), filename)
            f.read(mesh, True)
            self.generate_boundary_facets()
            self.subdomains = CellFunction("size_t", mesh)
            self.mesh = mesh
        elif filename[-4:] == ".xml":
            self._read_xml_mesh(filename)
        elif filename[-3:] == ".h5" or filename[-5:] == ".hdf5":
            self._read_hdf5_mesh(filename)
        else:
            raise SolverError('mesh or function space must specified to construct solver object')

        plot(self.boundary_facets, "boundary facets colored by ID")

    def generate_function_space(self, periodic_boundary):
        if 'element_degree' in self.settings:
            self.degree = self.settings['element_degree']
        else:
            self.degree = 1
        try:
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

    def translate_value(self, value, function_space = None):
        # for both internal and boundary values
        _degree = self.degree
        if function_space:
            W = function_space
        else:
            W = self.function_space
        if isinstance(value, (tuple,)):  # FIXME: json dump tuple into list, 
            if len(value) >= self.dimension and isinstance(value[0], (numbers.Number)):
                values_0 = Constant(value)
            elif len(value) >= self.dimension and isinstance(value[0], (str)):
                values_0 = interpolate(Expression(value, degree = _degree), W)
            else:
                raise TypeError(' {} is supplied, but only tuple of number and string expr are supported'.format(type(value)))
        elif  isinstance(value, (list, np.ndarray)) and len(value) > self.current_step:
            values_0 = value[self.current_step]
        elif  isinstance(value, (numbers.Number)):
            values_0 = Constant(value)
        elif isinstance(value, (Constant, Function)):  # CellFunction isinstance of Function???
            values_0 = value  # leave it as it is, since they can be used in equation
        elif isinstance(value, (Expression, )): 
            values_0 = interpolate(Expression(value, degree = _degree), W)
        elif callable(value):
            values_0 = value(self.current_time)
        elif isinstance(value, (str, )):
            if os.path.exists(value):
                # also possible continue from existent solution, or interpolate from diff mesh density
                values_0 = Function(W)
                File(value) >> values_0
                #project(velocity, self.vector_space)  # FIXME: diff element degree is not tested
                import fenicstools
                values_0 = fenicstools.interpolate_nonmatching_mesh(values_0 , W)
            else:  # C++ expressing string
                values_0 = interpolate(Expression(value, degree = _degree), W)
        elif type(value) == type(None):
            raise TypeError('None type is supplied')
        else:
            raise TypeError(' {} is supplied, not tuple, number, Constant,file name, Expression'.format(type(value)))
            #values_0 = None
        return values_0

    def get_boundary_value(self, bc, time_iter_=None):
        if self.transient_settings['transient']:
            if isinstance(bc['value'], list):  # if it is a list but not tuple
                bvalue = (bc['value'])[time_iter_] #already, interpolated function, should be
            elif callable(bc['value']): 
                bvalue = bc['value'](self.get_current_time(time_iter_))
            else:
                bvalue = bc['value']
        else:
            bvalue = bc['value']
        return self.translate_value(bvalue, time_iter_)

    def get_body_source(self):
        if isinstance(self.body_source, (dict)):  # a dict of subdomain, perhaps easier by giving an Expression
            vdict = copy.copy(self.body_source)
            for k in vdict:
                vdict[k]['value'] = self.translate_value(self.body_source[k]['value'])
            return vdict
        else:
            return self.translate_value(self.body_source)

    def get_time_step(self, time_iter_):
        ## fixed step, but could be supplied with an np.array/list
        try:
            dt = float(self.transient_settings['time_step'])
        except:
            ts = self.transient_settings['time_series']
            if len(ts) >= time_iter_:
                dt = ts[time_iter_] - ts[time_iter_]
            else:
                print('time step can only be a sequence or scaler')
        #self.mesh.hmin()  # Compute minimum cell diameter. courant number
        return dt

    def get_current_time(self, time_iter_):
        try:
            dt = float(self.transient_settings['time_step'])
            tp = self.transient_settings['starting_time'] + dt * (time_iter_ - 1)
        except:
            if len(self.transient_settings['time_series']) >= time_iter_:
                tp = self.transient_settings['time_series'][time_iter_]
            else:
                print('time point can only be a sequence of time series or derived from constant time step')
        return tp

    def solve_transient(self):
        #
        trial_function = TrialFunction(self.function_space)
        test_function = TestFunction(self.function_space)
        # Define functions for transient loop
        up_current = self.get_internal_field()  # init to default or user provided constant
        up_prev = Function(self.function_space)
        up_prev.assign(up_current)
        ts = self.transient_settings

        # Define a parameters for a stationary loop
        self.current_time = ts['starting_time']
        self.current_step = 0
        if ts['transient']:
            t_end = ts['ending_time']
        else:
            t_end = self.current_time+ 1

        #print(ts, self.current_time, t_end)
        # Transient loop also works for steady, by set `t_end = self.time_step`
        timer_solver_all = Timer("TimerSolveAll")
        timer_solver_all.start()
        while (self.current_time < t_end):
            if ts['transient']:
                dt = self.get_time_step(self.current_step)
            else:
                dt = 1

            ## overloaded by derived classes, maybe move out of temporal loop if boundary does not change form
            # only NS equation needs current value to build form
            F, Dirichlet_bcs_up = self.generate_form(self.current_step, trial_function, test_function, up_current, up_prev)

            up_prev.assign(up_current)  #
            up_current = self.solve_static(F, up_current, Dirichlet_bcs_up)  # solve for each time step, up_prev tis not needed
            #plot(up_current, title = "Value at time: " + str(t))
            print("Current time = ", self.current_time, " TimerSolveAll = ", timer_solver_all.elapsed())
            # stop for steady case, or update time
            if not self.transient_settings['transient']:
                break
            self.current_step += 1
            self.current_time += dt
        ## end of time loop
        timer_solver_all.stop()

        return up_current

    ####################################
    def solve_linear_problem(self, F, u, Dirichlet_bcs):
        """
        a_T, L_T = system(F)
        A_T = assemble(a_T)

        b_T = assemble(L_T)
        #for bc in bcs: print(type(bc))
        [bc.apply(A_T, b_T) for bc in bcs]  # apply Dirichlet BC
        solver = 
        self.set_solver_parameters(solver)

        solver.solve(A_T, T.vector(), b_T)
        """
        problem = LinearVariationalProblem(lhs(F), rhs(F), u, Dirichlet_bcs)
        solver = LinearVariationalSolver(problem)
        self.set_solver_parameters(solver)

        solver.solve()
        return u

    def solve_nonlinear_problem(self, F, u_current, Dirichlet_bcs, J):
        problem = NonlinearVariationalProblem(F, u_current, Dirichlet_bcs, J)
        solver = NonlinearVariationalSolver(problem)
        self.set_solver_parameters(solver)

        solver.solve()
        return u_current

    def set_solver_parameters(self, solver):
        # Define a dolfin linear algobra solver parameters
        if dolfin.MPI.size(dolfin.mpi_comm_world())>1:
            using_MPI = True
        else:
            using_MPI = False

        parameters["linear_algebra_backend"] = "PETSc"  #UMFPACK: out of memory, PETSc divergent
        #parameters["linear_algebra_backend"] = "Eigen"  # 'uBLAS' is not supported any longer

        parameters["mesh_partitioner"] = "SCOTCH"
        #parameters["form_compiler"]["representation"] = "quadrature"
        parameters["form_compiler"]["optimize"] = True
        """
        #solver.parameters["linear_solver"] = 'default'
        #solver.parameters["preconditioner"] = 'default'
        if using_MPI:
            #parameters['linear_solver'] = 'bicgstab'  # "gmres" # not usable in MPI
            parameters['preconditioner']= "hypre_euclid"
        else:
            #parameters['linear_solver'] = 'default'  # is not a parameter for LinearProblemSolver
            parameters['preconditioner'] = "default"  # 'default', ilu only works in serial
        """
        """
        for key in self.solver_settings['solver_parameters']:
            solver.parameters[key] = self.solver_settings['solver_parameters'][key]
        #param = self.solver_settings['solver_parameters']
        # these are only for iterative solver, the default solver, lusolver, neeed not such setttings
        #solver.parameters["relative_tolerance"] = param["relative_tolerance"] 
        #solver.parameters["maximum_iterations"] = param["maximum_iterations"]
        #solver.parameters["monitor_convergence"] = param["monitor_convergence"]
        """

    def solve_amg(self, F, u, bcs):
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