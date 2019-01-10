# -*- coding: utf-8 -*-
# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2018 - Qingfeng Xia <qingfeng.xia iesensor.com>         *
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

"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function, division
import math
import copy
import numpy as np

from dolfin import *
from mshr import *
from FenicsSolver import SolverBase

from config import is_interactive
interactively = is_interactive()

using_fenics_solver = True
using_segegrate_solver = not using_fenics_solver

t_end = 5.0            # final time
num_steps = 5000   # number of time steps
dt = t_end / num_steps # time step size
mu = 0.01         # dynamic viscosity
rho = 1            # density
transient_settings = {'transient': False, 'starting_time': 0.0, 'time_step': dt, 'ending_time': t_end}

# Create mesh
x_min, x_max = 0, 2.2
y_min, y_max = 0, 0.41
c_x, c_y = 0.2, 0.2
c_radius = 0.05
channel = Rectangle(Point(x_min, y_min), Point(x_max, y_max))
cylinder = Circle(Point(c_x, c_y), c_radius)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

vel_mean = 4.0
# Define inflow profile
inflow_profile = ('{}*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)'.format(vel_mean), '0')

if using_fenics_solver:
    coupling_energy_equation = False
    mu = 3
    rho = 100
    #vel_mean = 0.4

    length_scale = (y_max - y_min) 
    Re  = length_scale * vel_mean / (mu/rho)
    print("Reynolds number = ", Re)

    gdim = 2
    zero_vel = Constant(gdim*(0,))
    p_outlet = 10000
    T_ambient = 300 

    _inlet_vel = Constant((vel_mean, 0)) # Expression(inflow_profile, degree = 2)

    inlet_boundary   = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], x_min))
    outlet_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], x_max))
    static_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], y_min) or near(x[1], y_max))
    cylinder_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and x[0]>0.1 and x[0]<0.3 and x[1]>0.1 and x[1]<0.3)

    from collections import OrderedDict
    bcs = OrderedDict()

    bcs["wall"] = {'boundary': static_boundary, 'boundary_id': 0, 
            'values':[{'variable': "velocity",'type': 'Dirichlet', 'value': zero_vel, 'unit': 'm/s'},
                             {'variable': "temperature",'type': 'Dirichlet', 'value': T_ambient}   ]}
    bcs["cylinder"] = {'boundary': cylinder_boundary, 'boundary_id': 1, 
            'values':[{'variable': "velocity",'type': 'Dirichlet', 'value': zero_vel, 'unit': 'm/s'},
                             {'variable': "temperature",'type': 'Dirichlet', 'value': T_ambient}   ]}
    bcs["inlet"] = {'boundary': inlet_boundary, 'boundary_id': 2, 
                                    'values':[{'variable': "velocity", 'type': 'Dirichlet', 'value': _inlet_vel},
                                    {'variable': "temperature",'type': 'Dirichlet', 'value': T_ambient} ]}
    bcs["outlet"] = {'boundary': outlet_boundary, 'boundary_id': 3, 
                                    'values':[{'variable': "pressure", 'type': 'Dirichlet', 'value': p_outlet},
                                                    {'variable': "temperature",'type': 'Dirichlet', 'value': T_ambient} ]}


    s = copy.copy(SolverBase.default_case_settings)
    s['mesh'] = mesh
    s['fe_family'] = 'CG'  # 'P'
    print(info(mesh))
    s['boundary_conditions'] = bcs
    s['initial_values'] = {'velocity': (0, 0), "temperature": T_ambient, "pressure": p_outlet}
    s['solver_settings']['reference_values'] = {'velocity': (1, 1), "temperature": T_ambient, "pressure": p_outlet}
    s['solver_settings']['solver_parameters'] = {
                                    "relative_tolerance": 1e-9,  # mapping to solver.parameters of Fenics
                                    "maximum_iterations": 50,
                                    "relaximation_parameter": 0.001,
                                    "monitor_convergence": True,  # print to console
                                    },
    s['solving_temperature'] = coupling_energy_equation
    fluid = {'name': 'oil', 'kinematic_viscosity': mu/rho, 'density': rho, 
                'specific_heat_capacity': 4200, 'thermal_conductivity':  0.1, 'Newtonian': True}
    s['material'] = fluid

    #s['advection_settings'] = {'Re': Re, 'stabilization_method': 'G2' , 'kappa1': 4, 'kappa2': 2}

    from FenicsSolver import CoupledNavierStokesSolver
    solver = CoupledNavierStokesSolver.CoupledNavierStokesSolver(s)  # set a very large viscosity for the large inlet width
    solver.using_nonlinear_solver = False
    solver.transient_settings['transient']  = False
    from pprint import pprint
    #pprint(solver.settings)
    plot(solver.boundary_facets)
    up0 = solver.solve()  # get a realistic intial valve from static simulation
    solver.initial_values = up0
    solver.transient_settings['transient'] = transient_settings
    solver.solve()  # no need to transform static bc into transient bc

    if interactively:
        solver.plot()

if using_segegrate_solver:
    # Define expressions used in variational forms
    f  = Constant((0, 0))
    k  = Constant(dt)
    mu = Constant(mu)
    rho = Constant(rho)

    # Define boundaries
    inflow   = 'near(x[0], {})'.format(x_min)
    outflow  = 'near(x[0], {})'.format(x_max)
    walls    = 'near(x[1], {}) || near(x[1], {})'.format(y_min, y_max)
    cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
    #AutoSubDomain

    # Define inflow profile
    inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

    # Define function spaces, why not CG ?
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    
    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)


    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)

    U  = 0.5*(u_n + u)
    n  = FacetNormal(mesh)

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))

    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx \
       + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
       + inner(sigma(U, p_n), epsilon(v))*dx \
       + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
       - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    # Create XDMF files for visualization output
    xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
    xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

    # Create time series (for use in reaction_system.py)
    timeseries_u = TimeSeries('navier_stokes_cylinder/velocity_series')
    timeseries_p = TimeSeries('navier_stokes_cylinder/pressure_series')

    # Save mesh to file (for use in reaction_system.py)
    File('navier_stokes_cylinder/cylinder.xml.gz') << mesh

    # Create progress bar
    progress = Progress('Time-stepping')
    set_log_level(PROGRESS)

    # Time-stepping
    t = 0
    for n in range(num_steps):

        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, u_.vector(), b3, 'cg', 'sor')

        # Plot solution
        plot(u_, title='Velocity')
        plot(p_, title='Pressure')

        # Save solution to file (XDMF/HDF5)
        xdmffile_u.write(u_, t)
        xdmffile_p.write(p_, t)

        # Save nodal values to file
        timeseries_u.store(u_.vector(), t)
        timeseries_p.store(p_.vector(), t)

        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)

        # Update progress bar
        progress.update(t / t_end)
        print('u max:', u_.vector().array().max())

    # Hold plot
    interactive()
