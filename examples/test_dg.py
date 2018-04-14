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
from mshr import *

def defined(x):
    return x in locals() or x in globals()

""" pin on disc tribology heat transfer, 3D quasi-static simulation
# see paper:  Qingfeng Xia, D.G., Andrew Owen, Gervas Franceschini. 
Quasi-static thermal modelling of multi-scale sliding contact for unlubricated brush seal materials.
in Proceedings of ASME Turbo Expo 2018: Power for Land, Sea and Air. 2018. Lillestrom (Oslo), Norway.
"""

using_3D = False
using_simple_geomtry= True and using_3D
fe_degree = 1 # 2 is possible but use more memory
using_DG_solver = False
has_convection = True
omega = 0.5  # angular velocity 0.005 is correct without compensation, 
#but SPUG still got error
interactively = True
set_log_level(ERROR)

parameters["ghost_mode"] = "shared_facet"

r_p = 2.5e-2
thickness_p = 0.01
r_d = 1.5e-1
thickness_d = 1e-2
eccentricity = 1e-1  # track position

z_original = 0
disc_zmin, disc_zmax = z_original-thickness_d, z_original
pin_zmin, pin_zmax = z_original, z_original + thickness_p
#heat_source_zmin, heat_source_zmax = z_original, z_original+heat_source_depth

pin_pos = Point(0,eccentricity,pin_zmin)  # theta = 0
pin_pos_top = Point(0, eccentricity, pin_zmax)

if using_simple_geomtry:
    res = int(10 / fe_degree)
    mesh = UnitCubeMesh(res*2,res*2,res)
    bpin = AutoSubDomain(lambda x, on_boundary:  near(x[2], 1) and x[0]>0.5 and x[1]>0.5 and on_boundary)
    bdisc = AutoSubDomain(lambda x, on_boundary: near(x[2], 0) and on_boundary)
else:
    if using_3D:
        geom_disc = Cylinder(Point(0,0, disc_zmax), Point(0,0, disc_zmin), r_d, r_d) # disc height in Z- direction, z<0
        geom_pin = Cylinder(pin_pos, pin_pos_top, r_p, r_p)
        bpin = AutoSubDomain(lambda x, on_boundary:  x[2]>z_original and on_boundary)
        bdisc = AutoSubDomain(lambda x, on_boundary: (near(x[2], disc_zmin)) and on_boundary)
    else:
        geom_disc = Circle(Point(0,0, disc_zmax), r_d)
        geom_pin = Rectangle(Point(-r_p, r_d*0.98, 0), Point(r_p, r_d + thickness_p, 0))
        bpin = AutoSubDomain(lambda x, on_boundary: near(x[1], r_d + thickness_p) and on_boundary)
        bdisc = AutoSubDomain(lambda x, on_boundary: (x[1]<r_d) and on_boundary)
    geom = CSGUnion(geom_disc, geom_pin)
    mesh = generate_mesh(geom, 60)

#bpin = AutoSubDomain(lambda x, on_boundary:  x[2]>z_original and on_boundary)
#bdisc = AutoSubDomain(lambda x, on_boundary: (near(x[2], z_original) or x[2]<z_original) and on_boundary)
#################################################

T_hot = 360
T_cold = 300
T_ambient = 300
boundary_id_disc = 1
boundary_id_pin = 2
htc =200
intensity = 1e7


velocity_code_3D = '''

class Velocity : public Expression
{
public:

  // Create expression with any components
  Velocity() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const double x0 = %f;
    const double y0 = %f;
    const double omega = %f;

    double r = sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0));
    double v = omega * r;
    double a = atan2((x[1]-y0), (x[0]-x0));
    values[0] = -v * sin(a);
    values[1] = v * cos(a);
    values[2] = 0.0;
  }
};
'''%(0, 0, omega)

velocity_code_2D = '''

class Velocity : public Expression
{
public:

  // Create expression with any components
  Velocity() : Expression(2) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const double x0 = %f;
    const double y0 = %f;
    const double omega = %f;
    const double r_d = %f;

    double r = sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0));
    double v = omega * r;
    double a = atan2((x[1]-y0), (x[0]-x0));
    if (r<= r_d) {
        values[0] = -v * sin(a);
        values[1] = v * cos(a);
    } else {
        values[0] = 0;
        values[1] = 0;
    }
  }
};
'''%(0, 0, omega, r_d)

source_code_2D = '''

class Source : public Expression
{
public:

  // Create expression with any components
  Source() : Expression() {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const double x0 = %f;
    const double y0 = %f;
    const double intensity = %f;
    const double r_d = %f;

    double r = sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0));
    if (r>= r_d) {
        values[0] = intensity;
    } else {
        values[0] = 0;
    }
  }
};
'''%(0, 0, intensity, r_d)

vector_degree = fe_degree+1
if using_3D:
    v_e = Expression(cppcode=velocity_code_3D, degree=vector_degree)
    if using_simple_geomtry:
        v_e = Expression(("1.0", "-1", "-1"), degree=vector_degree)
else:
    v_e = Expression(cppcode=velocity_code_2D, degree=vector_degree)
    #v_e = Expression(("-1.0", "-1"), degree=vector_degree)
    s_e = Expression(cppcode=source_code_2D, degree=fe_degree)

##################################################
bcs = { 
        "hot": {'boundary': bpin, 'boundary_id': boundary_id_pin, 'values': {
                    #'temperature': {'variable': 'temperature', 'type': 'HTC', 'value': Constant(htc), 'ambient': Constant(T_ambient)}
                    'temperature': {'variable': 'temperature', 'type': 'Neumann', 'value':0}
                 } },
        "cold": {'boundary': bdisc, 'boundary_id': boundary_id_disc, 'values': {
                    'temperature': {'variable': 'temperature', 'type': 'HTC', 'value': Constant(htc), 'ambient': Constant(T_ambient)}
                    #'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_hot-50)}
                 } },
}

settings = {'solver_name': 'ScalerEquationSolver',
                'mesh': mesh, 'periodic_boundary': None, 'fe_degree': fe_degree,
                'boundary_conditions': bcs,
                'body_source': None,
                'initial_values': {'temperature': T_ambient},
                'material':{'density': 1000, 'specific_heat_capacity': 4200, 'thermal_conductivity': 15}, 
                'solver_settings': {
                    'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.1, 'ending_time': 1},
                    'reference_values': {'temperature': T_ambient},
                    'solver_parameters': {"relative_tolerance": 1e-11,  # mapping to solver.parameters of Fenics
                                                    "maximum_iterations": 500,
                                                    "monitor_convergence": True,  # print to console
                                                    },
                    },
                # solver specific settings
                'scaler_name': 'temperature',
                }


if using_DG_solver:
    from FenicsSolver.ScalerEquationDGSolver import ScalerEquationDGSolver
    solver = ScalerEquationDGSolver(settings)
    velocity = interpolate(v_e, solver.vector_function_space)
else:
    from FenicsSolver.ScalerEquationSolver import ScalerEquationSolver
    solver = ScalerEquationSolver(settings)
    V = VectorFunctionSpace(solver.mesh, 'CG', vector_degree)
    velocity = interpolate(v_e, V)

plot(mesh, title="mesh")
#plot(solver.boundary_facets, title="boundary facets colored by ID")  # matplotlib can not plot 1D
plot(velocity, title="velocity")
if has_convection:
    solver.convective_velocity = velocity

#DG for conductivity and heat source
Q = solver.function_space
DG0 = FunctionSpace(solver.mesh, 'DG', 0)  # can not mixed higher order CG and DG?
solver.body_source = interpolate(s_e, DG0)
#plot(solver.body_source )

T = solver.solve()

dx= Measure("dx", subdomain_data=solver.subdomains)
heat1 = assemble(solver.body_source*dx)
print('heat generated: ', heat1)

ds= Measure("ds", subdomain_data=solver.boundary_facets)
surface_normal = FacetNormal(solver.mesh)
m = solver.material
cooling_disc = assemble(htc*(T-T_ambient)*ds(boundary_id_disc))
cooling_pin = assemble(htc*(T-T_ambient)*ds(boundary_id_pin))
print('convective cooling from disc and pin are', cooling_disc, cooling_pin)
Q_disc = assemble(dot(surface_normal, velocity) * T *\
        Constant(m['density']*m['specific_heat_capacity'])*ds(boundary_id_disc))  #-Constant(T_ambient)
print("mass heat flow out of system from disc boundary is:", Q_disc)
Q_pin = assemble(dot(surface_normal, velocity) * T *\
        Constant(m['density']*m['specific_heat_capacity'])*ds(boundary_id_pin))
print("mass heat flow out of system from pin boundary is:", Q_pin)

T = solver.plot()
if interactively:
    interactive()