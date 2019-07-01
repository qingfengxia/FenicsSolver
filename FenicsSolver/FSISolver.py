# not yet fully tested code,
# copyright qingfeng Xia 2018

"""

#How it works:
1.  provide mesh with fluid and solid submeshes, 
2. coupled solver will detect matching interfaces, modify solver setting (boundary condition)
3. create participant solvers (as a ordered list)
4. solve the fluid first, , setup solid boundary condition (traction force) for solid solver, no need to move mesh for solid solver
5. move mesh for fluid solver,  w_current, w_prev need to be in different function space, in order to save deformed mesh and data

Limitations:
- serial mapping only
- no movement relaxation
- no higher Re fluid solver
- will not support multiple frictional contact

"""

from __future__ import print_function, division, absolute_import
from FenicsSolver.CoupledNavierStokesSolver import CoupledNavierStokesSolver
from FenicsSolver.LargeDeformationSolver import LargeDeformationSolver
from FenicsSolver.LinearElasticitySolver import LinearElasticitySolver
from FenicsSolver.SolverBase import SolverBase, SolverError
from dolfin import *
import math, copy
import numpy as  np

_debug = False

class CoupledSolver():
    """ This CoupledSolver class provide a skeleton for coupling sovler
    it contains a list of solver, and coordiate solvers in overrided `solve_current_step()`
    it can be used for sequential coupling, FSI, mixed-dimension solver
    """
    def __init__(self, solver_input):
        self.settings = solver_input

    def solve_transient(self):
        #
        self.init_solver()
        file = File("pressure_output.pvd", "compressed")

        # Define a parameters for a stationary loop
        self.transient_settings = self.settings['transient_settings']
        ts = self.settings['transient_settings']
        self.current_time = ts['starting_time']
        self.current_step = 0
        if ts['transient']:
            t_end = ts['ending_time']
        else:
            t_end = self.current_time+ 1

        cs = self.settings['coupling_settings']
        #print(ts, self.current_time, t_end)
        # Transient loop also works for steady, by set `t_end = self.time_step`
        timer_solver_all = Timer("TimerSolveAll")  # 2017.2 Ubuntu Python2 errors
        timer_solver_all.start()
        while (self.current_time < t_end):
            if ts['transient']:
                dt = self.get_time_step(self.current_step)
            else:
                dt = 1

            # in the first step, initial flow field will be calc in a steady way, since up_current == up_prev
            for s in self.solver_list:
                s.current_step = self.current_step
            ## overloaded by derived classes, maybe move out of temporal loop if boundary does not change form
            self.solve_current_step()
            p_result = self.fluid_solver.w_current.split()[1]
            p_result.rename('pressure', 'label')
            file << (p_result, self.current_time)  # todo: moved to fluid_solver
            u_result = self.fluid_solver.w_current.split()[0]
            u_result.rename('velocity', 'label')
            file << (u_result, self.current_time)  # todo: moved to fluid_solver

            print("Current time = ", self.current_time, " TimerSolveAll = ", timer_solver_all.elapsed())
            # stop for steady case, or update time

            if not self.transient_settings['transient']:
                break
            #quasi-static, check value change is small enough!
            
            self.current_step += 1
            self.current_time += dt
        ## end of time loop
        timer_solver_all.stop()
        self.plot_result()

        return [solver.result for solver in self.solver_list]

    def init_solver(self):
        for solver in self.solver_list:
            solver.init_solver()

    def solve(self):
        self.result = self.solve_transient()
        return self.result

    def plot_result(self):
        for solver in self.solver_list:
            solver.plot()

    def save(self):
        pass

    def get_time_step(self, time_iter_):
        ## fixed step, but could be supplied with an np.array/list
        try:
            dt = float(self.transient_settings['time_step'])
        except:
            ts = self.transient_settings['time_series']
            if len(ts) >= time_iter_:
                dt = ts[time_iter_] - ts[time_iter_]
            else:
                print('time step can only be a sequence or scalar')
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


class FSISolver(CoupledSolver):
    """ 
    """
    def __init__(self, solver_input):
        self.settings = solver_input
        for s in self.settings['participants']:
            if s['solver_domain'] == "fluidic":
                self.fluid_solver = CoupledNavierStokesSolver(s['settings'])
            elif s['solver_domain'] == "elastic":
                #self.solid_solver = LargeDeformationSolver(s['settings'])
                self.solid_solver =LinearElasticitySolver(s['settings'])
            else:
                raise SolverError("unsupported subdomain solver: {}".format(s['solver_name']))
        self.solver_list = [self.fluid_solver, self.solid_solver]
        self.detect_interfaces()
        self.original_solid_mesh = copy.copy(self.solid_solver.mesh)
        self.original_fluid_mesh = copy.copy(self.fluid_solver.mesh)
    
        self.using_submesh = True        #submesh method, topo does NOT change with mesh moving?
        self.parent_mesh = self.settings['parent_mesh']
        assert self.fluid_solver.settings['fe_degree']+1 == self.solid_solver.settings['fe_degree']
        self.detect_interface_mapping()

        #self.parent_vector_function_space = VectorFunctionSpace(self.parent, 
        #                self.fluid_solver.settings['fe_family'], self.fluid_solver.settings['fe_degree'])
        #self.parent_facet_function = MeshFunction('double', self.parent_vector_function_space.mesh(), mesh.topology().dim() - 1)

        #self.solid_facet_function = FacetFunction('double', self.solid_vector_function_space)
        #self.fluid_facet_function = FacetFunction('double', self.fluid_vector_function_space)


        #self.solid_boundary = BoundaryMesh(self.solid_solver.mesh, "exterior") # instead of FacetMesh for better performance
        #self.fluid_boundary = BoundaryMesh(self.fluid_solver.mesh, "exterior")
        #from domain to boundary solid_boundary_f.interpolate(cellFunction)
        self.original_fb_vector_fs = VectorFunctionSpace(self.original_fluid_mesh, 
                        self.fluid_solver.settings['fe_family'], self.fluid_solver.settings['fe_degree'])
        self.original_sb_vector_fs = VectorFunctionSpace(self.original_solid_mesh,
                        self.solid_solver.settings['fe_family'], self.solid_solver.settings['fe_degree'])

        self.mesh_offset = Function(self.original_fb_vector_fs)
        self.previous_fluid_mesh_disp = Function(self.original_fb_vector_fs)

    #################### SubmeshMapper #################
    def detect_interface_mapping(self):
        #"For mixed FunctionSpaces vertex index is offset with the number of dofs per vertex" (this also applies for VectorFunctionSpaces). 
        #dof mapping
        self.solid_parent_vi = self.solid_solver.mesh.data().array('parent_vertex_indices', 0)
        self.fluid_parent_vi = self.fluid_solver.mesh.data().array('parent_vertex_indices', 0)
        #print(self.solid_parent_vi)
        #print(len(self.fluid_parent_vi), type(self.fluid_parent_vi))  # find out shared vertex
        self.interface_parent_vi = np.intersect1d(self.solid_parent_vi, self.fluid_parent_vi)
        vl = []
        for vi in self.interface_parent_vi:
            vl.append((np.nonzero(self.fluid_parent_vi == vi)[0][0], np.nonzero(self.solid_parent_vi == vi)[0][0]))
        self.interface_fluid_solid_vi = vl
        #print(self.interface_parent_vi, self.interface_fluid_solid_vi)

        # can be split into 2 function from here
        self.fluid_V1 = VectorFunctionSpace(self.fluid_solver.mesh,
                        self.fluid_solver.settings['fe_family'], 1)  # msut be 1  for vertex -> DoF mapping
        self.solid_V1 = VectorFunctionSpace(self.original_solid_mesh,
                        self.solid_solver.settings['fe_family'], 1)  # msut be 1  for vertex -> DoF mapping
        v2d = vertex_to_dof_map(self.solid_V1)
        self.solid_v2d = v2d.reshape((-1, self.solid_solver.dimension))
        #print('self.solid_v2d = ', self.solid_v2d)

        # for scalar, vertex = numpy array
        v2d = vertex_to_dof_map(self.fluid_V1)  #vector degree 1
        self.fluid_v2d = v2d.reshape((-1, self.fluid_solver.dimension))
        #print('self.fluid_v2d = ', self.fluid_v2d)

        self.fluid_T1 = TensorFunctionSpace(self.fluid_solver.mesh,
                        self.fluid_solver.settings['fe_family'], 1)  # msut be 1  for vertex -> DoF mapping
        self.solid_T1 = TensorFunctionSpace(self.original_solid_mesh,
                        self.solid_solver.settings['fe_family'], 1)  # msut be 1  for vertex -> DoF mapping
        v2d = vertex_to_dof_map(self.fluid_T1)
        self.fluid_v2d_tensor = v2d.reshape((-1, self.fluid_solver.dimension* self.fluid_solver.dimension))

        v2d = vertex_to_dof_map(self.solid_T1)
        #print('vertex_to_dof_map(self.solid_T1) = ', v2d.shape, v2d[:12])
        self.solid_v2d_tensor = v2d.reshape((-1, self.solid_solver.dimension*self.solid_solver.dimension))
        #print('self.solid_v2d_tensor = ', self.solid_v2d_tensor.shape, self.solid_v2d_tensor)

        #set(self.solid_parent_vi).intersection(set(self.fluid_parent_vi));   np.array(list( a_py_set))  # not efficient

    def map_solid_to_fluid_vector(self, solid_f, target_space):
        assert self.using_submesh
        #
        solid_V1_temp = project(solid_f, self.solid_V1)
        fluid_V1_temp = Function(self.fluid_V1)  # set all DOF to zero?
        for fi, si in self.interface_fluid_solid_vi:
            fluid_V1_temp.vector()[self.fluid_v2d[fi]] = solid_V1_temp.vector()[self.solid_v2d[si]]
        return project(fluid_V1_temp, target_space)

    def map_fluid_to_solid_vector(self, fluid_f, target_space):
        assert self.using_submesh
        #rank 1 vector function
        fluid_V1_temp = project(fluid_f, self.fluid_V1)
        solid_V1_temp = Function(self.solid_V1)
        #print(self.fluid_v2d[0], fluid_V1_temp.vector()[self.fluid_v2d[0]])
        for fi, si in self.interface_fluid_solid_vi:
            #print(fi, fluid_V1_temp.vector()[self.fluid_v2d[fi]])
            solid_V1_temp.vector()[self.solid_v2d[si]] = fluid_V1_temp.vector()[self.fluid_v2d[fi]]
        return project(solid_V1_temp, target_space)

    def map_fluid_to_solid_tensor(self, sigma):
        #print( sigma.vector().get_local().shape)  1D  Petsc vector, size = npoint * dim * dim
        boundary_stress = Function(self.solid_T1)  # set all DOF to zero?
        for fi, si in self.interface_fluid_solid_vi:
            #print(fi, type(sigma.vector()[self.fluid_v2d_tensor[fi]]), sigma.vector()[self.fluid_v2d_tensor[fi]])
            # reverse stress sensor from fluid to solid
            boundary_stress.vector()[self.solid_v2d_tensor[si]] =  - sigma.vector()[self.fluid_v2d_tensor[fi]]
        return boundary_stress


    def solve_current_step(self):
        # only NS equation needs current value to build form
        self.fluid_solver.solve_current_step()
        if _debug:
            plot(self.fluid_solver.boundary_facets, title = "fluid boundary")
            plot(self.solid_solver.boundary_facets, title = "solid boundary")
            self.fluid_solver.plot()

        # self.up_trial_function, self.up_test_function, self.up_current, self.up_prev
        self.update_solid_interface(self.fluid_solver.w_current)  # set solid boundary_conditions

        self.solid_solver.solve_current_step()
        if _debug:
            self.solid_solver.plot()

        #move fluid mesh and set the boundary
        mesh_disp = self.update_fluid_interface(self.solid_solver.w_current)  # set solid boundary_conditions, 
        self.move_fluid_interface(mesh_disp)
        # self.move_solid_interface()  # not necessary for submeshing no interpolation

    def detect_interfaces(self, specific_type = 'FSI'):
        # matching by boundary name, not by coordinate coincidence, also comes from setting dict
        self.interfaces = {} # list of tuple of dict
        for key, bc in self.fluid_solver.settings['boundary_conditions'].items():
            if 'coupling' in bc and bc['coupling'] == specific_type:
                if key in self.solid_solver.settings['boundary_conditions']:
                    self.interfaces[key] = (bc, self.solid_solver.settings['boundary_conditions'][key])
                else:
                    raise SolverError('couplng boundary named `{}` in fluid_solver has no corresponding in solid_solver'.format(key))
        assert self.interfaces, 'interfaces dict should not be empty'

    def update_solid_interface(self, up_current):
        # setup stress boundary on the solid domain
        sigma = self.fluid_solver.viscous_stress(up_current, self.fluid_T1)
        boundary_stress = self.map_fluid_to_solid_tensor(sigma)  # reverse direction
        #boundary_stress = Constant(((0, 0), (100, 0)))  # test passed
        for iface in self.interfaces:
            #bc_values = {'type': 'stress', 'value': boundary_stress}
            self.solid_solver.settings['boundary_conditions'][iface]['value'] = boundary_stress
            self.solid_solver.settings['boundary_conditions'][iface]['type'] = 'stress'
            print("updated interface:", self.solid_solver.settings['boundary_conditions'][iface])

    def move_fluid_interface(self, mesh_disp):
        # assing no fluid mesh topo change
        self.previous_fluid_mesh = copy.copy(self.fluid_solver.mesh)
        self.mesh_offset.vector()[:] = mesh_disp.vector().get_local() - self.previous_fluid_mesh_disp.vector().get_local()
        ALE.move(self.fluid_solver.mesh, self.mesh_offset)
        self.previous_fluid_mesh_disp = mesh_disp

        #no need to fluid redefine function space, but mesh will not move in result file
        self.fluid_solver.update_solver_function_space(None)
        #
        #no need to move the solid mesh, for spatial interpolation if not using submesh mapping

    def generate_mesh_deformation_bc(self, V, bfunc):
        # does not mater for displacement velocity
        Dirichlet_bcs = []
        zero_vector = Constant(self.fluid_solver.dimension*(0.0,))
        for key, boundary in self.fluid_solver.boundary_conditions.items():
            if 'coupling' in boundary and boundary['coupling'] == 'FSI':
                dbc = DirichletBC(V, bfunc, self.fluid_solver.boundary_facets, boundary['boundary_id'])
            else:
                dbc = DirichletBC(V, zero_vector, self.fluid_solver.boundary_facets, boundary['boundary_id'])
            Dirichlet_bcs.append(dbc)
        return Dirichlet_bcs

    def update_fluid_interface(self, uv_current):
        # can NOT be shared between tume step
        deforming_from_original_mesh = True
        if deforming_from_original_mesh:
            disp, vel = self.solid_solver.displacement(), self.solid_solver.velocity()
            # move to __init__
            Vf = VectorFunctionSpace(self.original_fluid_mesh, self.fluid_solver.settings['fe_family'], self.fluid_solver.settings['fe_degree'])
            Vs = VectorFunctionSpace(self.original_solid_mesh, self.solid_solver.settings['fe_family'], self.solid_solver.settings['fe_degree'])

            #disp_bfunc = Function(Vf)#interpolate(disp, self.original_fb_vector_fs)  #
            #vel_bfunc = Function(Vf) #interpolate(vel, self.original_fb_vector_fs)
            disp_bfunc = self.map_solid_to_fluid_vector(disp, Vf)
            vel_bfunc = self.map_solid_to_fluid_vector(vel, Vf)
            if _debug:
                plot(vel, title = 'velocity_solid')
                plot(disp_bfunc, title = 'disp_bfunc_mapped_from_solid')
                plot(vel_bfunc, title = 'vel_bfunc_mapped_from_solid')

            #mapping to original mesh, from index matching diff subdomains?
            #build a single FacetMesh function, and set all the value
            bcs_displacement = self.generate_mesh_deformation_bc(Vf, disp_bfunc)
            # at the interface, solid_disp_current ; otherwise, zero
            bcs_velocity = self.generate_mesh_deformation_bc(Vf, vel_bfunc)
            # at the interface,  (solid_disp_current - solid_disp_prev) / dt, otherwise, zero
            mesh_disp, mesh_velocity = get_mesh_moving_displacement_and_velocity(Vf, self.original_fluid_mesh, bcs_displacement, bcs_velocity)

        else:  # incremental deformation from previous mesh
            Vs = VectorFunctionSpace(self.current_solid_mesh, self.solid_solver.settings['fe_family'], self.solid_solver.settings['fe_degree'])
            nDisp, nVel = Function(Vs), Function(Vs)
            nDisp.vector()[:] = disp.vector().get_local()
            nDisp.vector()[:] = disp.vector().get_local()
            disp_bfunc = project(nDisp, Vf)
            vel_bfunc = project(nVel, Vf)

            # array assignment from fluid solver current mesh to original mesh, if no topo change
            
            # mapping to original mesh, from index matching diff subdomains?
            #build a single FacetMesh function, and set all the value
            bcs_displacement = self.generate_mesh_deformation_bc(Vf, disp_bfunc)
            # at the interface, solid_disp_current ; otherwise, zero
            bcs_velocity = self.generate_mesh_deformation_bc(Vf, vel_bfunc)
            # at the interface,  (solid_disp_current - solid_disp_prev) / dt, otherwise, zero

            # deform from original mesh is only for small deformation
            # for large deformation, incremental mesh vel and disp
            mesh_disp, mesh_velocity = get_mesh_moving_displacement_and_velocity(V, self.original_fluid_mesh, bcs_displacement, bcs_velocity)

            # mesh after get mesh velocity?  # fluid mesh velocity, cna be get from du/dt
            self.previous_fluid_mesh = copy.copy(self.current_fluid_mesh)

        # set solid boundary_conditions, Dirichlet boundary for interface
        if _debug: plot(mesh_velocity, title = 'fluid_mesh_velocity')
        #interactive()
        self.fluid_solver.settings['reference_frame_settings']['mesh_velocity'] = mesh_velocity
        for iface in self.interfaces:
            #boundary_velocity = Constant((0.0001,0))  # FIXME: tmp test,  from cell to facet
            boundary_velocity = mesh_velocity
            bc_values = [{'variable': "velocity",'type': 'Dirichlet', 'value': boundary_velocity}]
            self.fluid_solver.settings['boundary_conditions'][iface]['value'] = bc_values

        print('max mesh disp', np.max(mesh_disp.vector().get_local()))
        return mesh_disp

    def move_solid_interface(self):
        disp = self.solid_solver.displacement()
        new_solid_mesh = copy.copy(self.original_solid_mesh)
        ALE.move(new_solid_mesh, project(disp, self.solid_V1))  # must match geometry degree 1
        self.current_solid_mesh = new_solid_mesh


###################
def get_mesh_moving_displacement_and_velocity(V, mesh, bcs_displacement, bcs_velocity):
    # see: chapter 4, 'Coupled Fluid-Structure Simulation of Flapping Wings', 2012
    # bcs: Dirichlet conditions for displacements
    # bcsp: Dirichlet conditions for velocities
    # return: internal displacement and  velocity
    #bfc = mesh.data().mesh_function('bfc')  # why, error
    
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(mesh.geometry().dim()*(0.0, ))  # source
    DG = FunctionSpace(mesh, "DG", 0)
    
    E = Function(DG)
    for c in cells(mesh):
        E.vector()[c.index()]=1./c.volume()  # set elastic modulus

    nu = 0.0  # invisicid
    mu = E /(2.0*(1.0  + nu))
    lmbda = E*nu /((1.0 + nu)*(1 - 2*nu))

    #domains = CellFunction("size_t", mesh)
    dx = Measure('dx', domain = mesh)
    #dx: Multiple domains found, making the choice of integration domain ambiguous.
    def sigma(v):
        return 2.0*mu*sym(grad(v)) + lmbda * tr(sym(grad(v)))* Identity(mesh.geometry().dim())
    a = inner(sigma(u), sym(grad(v)))*dx
    L = inner(f, v)*dx
    
    A = assemble(a)
    b = assemble(L)
    # set Dirichlet conditions for displacements
    [bc.apply(A, b) for bc in bcs_displacement]
    u = Function(V)
    # generalized minimal residual method(gmres) with ILU preconditioning(ilu)
    solve(A, u.vector(), b, 'gmres', 'ilu')

    # set Dirichlet conditions for velocities
    u_v = Function(V)
    [bc.apply(A, b)  for bc in bcs_velocity]
    solve(A, u_v.vector(), b, 'gmres', 'ilu')

    return u, u_v