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

transient = False
T_ambient =300
T_wall = 350
p_inlet = 1.1e5
p_outlet = 1e5

print("print dolfin.dolfin_version()", dolfin.dolfin_version())
length_scale = 1
max_vel=10 * length_scale

def setup(using_elbow = True, using_3D = False, compressible=False):
    from mshr import Box, Rectangle, generate_mesh
    zero_vel = Constant((0,0))
    if using_elbow:
        x_min, x_max = 0 * length_scale, 2*length_scale
        y_min, y_max = 0 * length_scale, 2*length_scale
        x_mid, y_mid = 1*length_scale, 1*length_scale
        if using_3D:
            dim = 3
            z_min, z_max = 0 * length_scale, 1*length_scale
            elbow = Box(Point(x_min, y_min, z_min), Point(x_mid,y_max, z_max)) + Box(Point(x_mid, y_mid, z_min), Point(x_max, y_max, z_max))
            mesh = generate_mesh(elbow, 20)
            front_back_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary \
                                               and (near(x[2], z_min) or near(x[2], z_max)))
            zero_vel = Constant((0, 0, 0))
        else:
            dim = 2
            elbow = Rectangle(Point(x_min, y_min), Point(x_mid,y_max)) + Rectangle(Point(x_mid, y_mid), Point(x_max, y_max))
            mesh = generate_mesh(elbow, 20)

        static_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary \
            and (near(x[0], x_min) or near(x[1], y_max) or near(x[0], x_mid) or near(x[1], y_mid)))
        bottom = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], y_min) )
        outlet = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], x_max))
    else:
        #length_scale = 1
        mesh = UnitSquareMesh(40, 100)
        static_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and (near(x[0], 0) or near(x[0], 1)))

        bottom = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 0) )
        outlet = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 1))
        #moving_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 1) )
        #bcs_u["moving"] = {'boundary': top, 'boundary_id': 2, 'variable': "velocity", 'type': 'Dirichlet', 'value': Constant((1,0))}

    from collections import OrderedDict
    bcs_u = OrderedDict()

    bcs_u["static"] = {'boundary': static_boundary, 'boundary_id': 1, 
            'values':[{'variable': "velocity",'type': 'Dirichlet', 'value': zero_vel, 'unit': 'm/s'},
                             {'variable': "temperature",'type': 'Dirichlet', 'value': T_wall}   ]}
    if using_3D:
        bcs_u["front_back"] = {'boundary': front_back_boundary, 'boundary_id': 5, 
            'values':[{'variable': "velocity",'type': 'Dirichlet', 'value': zero_vel},
                            {'variable': "temperature",'type': 'Dirichlet', 'value': T_wall}     ]}

    bcs_p = OrderedDict()
    bcs_p["outlet"] = {'boundary': outlet, 'boundary_id': 3, 
                                    'values':[{'variable': "pressure", 'type': 'Dirichlet', 'value': p_outlet},
                                                    {'variable': "temperature",'type': 'Dirichlet', 'value': T_ambient}     ]}

    #inlet_vel_expr = Expression('max_vel * (1.0 - pow(abs(x[0]-x_mid)/x_mid, 2))', max_vel=10, x_mid=0.5)
    x_c=0.5 * length_scale
    if using_3D:
        _init_vel = (0, max_vel*0.2, 0)
        class InletVelcotyExpression(Expression):
            def eval(self, value, x):
                value[0] = 0
                value[1] = max_vel * (1.0 - (abs(x[0]-x_c)/x_c)**2)
                value[2] = 0
            def value_shape(self):
                return (3,)
    else:
        _init_vel = (0, max_vel*0.2)
        class InletVelcotyExpression(Expression):
            def eval(self, value, x):
                value[0] = 0
                value[1] = max_vel * (1.0 - (abs(x[0]-x_c)/x_c)**2)
            def value_shape(self):
                return (2,)

    bcs_u["inlet"] = {'boundary': bottom, 'boundary_id': 2, 
                                    'values':[{'variable': "temperature",'type': 'Dirichlet', 'value': T_ambient} ]}
    if transient:
        vels = [Constant((0, max_vel)), InletVelcotyExpression(degree=1), InletVelcotyExpression(degree=1)]
        bcs_u["inlet"] ['values'].append({'variable': "velocity", 'type': 'Dirichlet', 'value': vels})
    else:
        vel = InletVelcotyExpression(degree=1)  # degree = 1, may not general enough, fixme
        #if compressible:
        #bcs_u["inlet"] ['values'].append({'variable': "pressure", 'type': 'Dirichlet', 'value': p_inlet})
        #else:
        bcs_u["inlet"] ['values'].append({'variable': "velocity", 'type': 'Dirichlet', 'value': vel})

    bcs = bcs_p.copy()
    for k in bcs_u:
        bcs[k] = bcs_u[k]

    import copy
    from FenicsSolver import SolverBase
    s = copy.copy(SolverBase.default_case_settings)
    s['mesh'] = mesh
    print(info(mesh))
    s['boundary_conditions'] = bcs
    s['initial_values'] = {'velocity': _init_vel, "temperature": T_ambient, "pressure": 1e5}

    return s

def test_compressible():
    s = setup(using_elbow = True, using_3D = True, compressible = True)
    fluid = {'name': 'ideal gas', 'kinematic_viscosity': 1e-2, 'density': 1.3}
    s['material'] = fluid

    from FenicsSolver import CompressibleNSSolver
    solver = CompressibleNSSolver.CompressibleNSSolver(s)  # set a very large viscosity for the large inlet width
    #solver.init_values = Expression(('1', '0', '1e-5'), degree=1)
    u,p, T= split(solver.solve())
    plot(u)
    plot(p)


def test_incompressible(using_elbow = True):
    s = setup(using_elbow, using_3D = False, compressible = False)
    solving_energy_equation = False

    if using_elbow:
        Re = 1e-3  # see pressure change,     # stabilization_method seems make no difference
    else:
        s['fe_degree'] = 10  # test failed, can not finish JIT
        Re = 10  # mesh is good enough to simulate higher Re

    fluid = {'name': 'oil', 'kinematic_viscosity': (length_scale * max_vel)/Re, 'density': 800}
    s['material'] = fluid

    s['advection_settings'] = {'Re': Re, 'stabilization_method': 'G2' , 'kappa1': 4, 'kappa2': 2}
    print("Reynolds number = ", Re)

    from FenicsSolver import CoupledNavierStokesSolver
    solver = CoupledNavierStokesSolver.CoupledNavierStokesSolver(s)  # set a very large viscosity for the large inlet width
    #solver.init_values = Expression(('1', '0', '1e-5'), degree=1)
    u,p= split(solver.solve())
    plot(u)
    plot(p)
    #plot(solver.viscous_heat(u,p))

    if solving_energy_equation:  # not fully tested
        from FenicsSolver import  ScalerEquationSolver
        solver_T = ScalerEquationSolver.ScalerEquationSolver(s)
        #Q = solver.function_space.sub(1)  # seem it is hard to share vel between. u is vectorFunction
        Q = solver_T.function_space
        cvel = VectorFunction(Q)
        cvel.interpolate(u)  # degree ? 
        selver_T.convective_velocity = cvel
        T = solver_T.solve()


if __name__ == '__main__':
    test_incompressible()
    test_incompressible(False)
    is_interactive = True
    if is_interactive:
        interactive()
    # manually call test function,  if discovered by google test, it will not plot in interactive mode
    #test_incompressible(False, True)  # Elbow 3D is slow but possible
    #test_compressible(True, True)