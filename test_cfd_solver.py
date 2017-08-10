from __future__ import print_function, division
import math
import numpy as np

from dolfin import *
import CoupledNavierStokesSolver
import SolverBase

def test():
    using_elbow = True
    if using_elbow:
        from mshr import Rectangle, generate_mesh
        elbow = Rectangle(Point(0, 0), Point(1,2)) + Rectangle(Point(1, 1), Point(2, 2))
        mesh = generate_mesh(elbow, 20)
        static_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary \
            and (near(x[0], 0) or near(x[1], 2) or near(x[0], 1) or near(x[1], 1) )
             )
        bottom = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 0) )
        outlet = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[0], 2))
    else:
        mesh = UnitSquareMesh(40, 100)
        static_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and (near(x[0], 0) or near(x[0], 1)))
        #moving_boundary = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 1) )
        bottom = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 0) )
        outlet = AutoSubDomain(lambda x, on_boundary: on_boundary and near(x[1], 1))

    # for enclosured place, no need to set pressure?

    from collections import OrderedDict
    bcs_u = OrderedDict()
    bcs_u["static"] = {'boundary': static_boundary, 'boundary_id': 1, 'variable': "velocity",'type': 'Dirichlet', 'value': Constant((0,0))}
    # cpp expr is not good as subclassing, which need
    #inlet_vel_expr = Expression('max_vel * (1.0 - pow(abs(x[0]-x_mid)/x_mid, 2))', max_vel=10, x_mid=0.5)
    max_vel=10
    x_mid=0.5
    class InletVelcotyExpression(Expression):
        def eval(self, value, x):
            value[0] = 0
            value[1] = max_vel * (1.0 - (abs(x[0]-x_mid)/x_mid)**2)
        def value_shape(self):
            return (2,)
    bcs_p = OrderedDict()
    bcs_p["outlet"] = {'boundary': outlet, 'boundary_id': 3, 'variable': "pressure", 'type': 'Dirichlet', 'value': Constant(1e5)}

    fluid = {'name': 'oil', 'kinetic_viscosity': 1e8, 'density': 1.3}

    solving_energy_equation = False
    transient = False
    if transient:
        vels = [Constant((0, 10)), InletVelcotyExpression(degree=1), InletVelcotyExpression(degree=1)]
        bcs_u["inlet"] = {'boundary': bottom, 'boundary_id': 2, 'variable': "pressure", 'type': 'Dirichlet', 'value': vels}
        solver = CoupledNavierStokesSolver(mesh, bcs_u, bcs_p, material = fluid, transient=True)  # set a very large viscosity for the large inlet width
    else:
        #bcs_u["inlet"] = {'boundary': bottom, 'boundary_id': 2, 'variable': "velocity", 'type': 'Dirichlet', 'value': Constant((0, 10))}
        #bcs_u["moving"] = {'boundary': top, 'boundary_id': 2, 'variable': "velocity", 'type': 'Dirichlet', 'value': Constant((1,0))}
        bcs_u["inlet"] = {'boundary': bottom, 'boundary_id': 2, 'variable': "velocity", 'type': 'Dirichlet', 'value': InletVelcotyExpression(degree=1)}

    import copy
    s = copy.copy(SolverBase.default_case_settings)
    s['mesh'] = mesh
    #s['pressure_boundary_conditions'] = bcs_p
    #s['velocity_boundary_conditions'] =  bcs_u
    bcs = bcs_p.copy()
    for k in bcs_u:
        bcs[k] = bcs_u[k]

    s['boundary_conditions'] = bcs
    s['material'] = fluid

    print("print dolfin.dolfin_version()", dolfin.dolfin_version())
    solver = CoupledNavierStokesSolver.CoupledNavierStokesSolver(s)  # set a very large viscosity for the large inlet width
    #solver.init_values = Expression(('1', '0', '1e-5'), degree=1)
    u,p= split(solver.solve())
    plot(u)
    plot(p)
    #plot(solver.viscous_heat(u,p))

    if solving_energy_equation:
        from . import  ScalerEquationSolver
        bcs_temperature = {}
        bcs_u["inlet"] = {'boundary': inlet, 'boundary_id': 2, 'type': 'Dirichlet', 'value': 350}
        solver_T = ScalerEquationSolver.ScalerEquationSolver(mesh, bcs_temperature)  # need update with new API
        #Q = solver.function_space.sub(1)  # seem it is hard to share vel between. u is vectorFunction
        Q = solver_T.function_space
        cvel = Function(Q)
        cvel.interpolate(u)
        selver_T.convective_velocity = cvel
        T = solver_T.solve()

    interactive()


if __name__ == '__main__':
    test()