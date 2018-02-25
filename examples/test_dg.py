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

# pin on disc, 3D quasi-static simulation
# see paper: 

r_p = 1.5e-3
thickness_p = 0.002
r_d = 1.5e-2
thickness_d = 1e-3
eccentricity = 1e-2  # track position

z_original = 0
disc_zmin, disc_zmax = z_original-thickness_d, z_original
pin_zmin, pin_zmax = z_original, z_original + thickness_p
#heat_source_zmin, heat_source_zmax = z_original, z_original+heat_source_depth

pin_pos = Point(0,eccentricity,pin_zmin)  # theta = 0
pin_pos_top = Point(0, eccentricity, pin_zmax)

geom_disc = Cylinder(Point(0,0, disc_zmax), Point(0,0, disc_zmin), r_d, r_d) # disc height in Z- direction, z<0
geom_pin = Cylinder(pin_pos, pin_pos_top, r_p, r_p)
geom = CSGUnion(geom_disc, geom_pin)
mesh = generate_mesh(geom, 60)

parameters["ghost_mode"] = "shared_facet"
bpin = AutoSubDomain(lambda x, on_boundary:  x[2]>z_original and on_boundary)
bdisc = AutoSubDomain(lambda x, on_boundary: (near(x[2], z_original) or x[2]<z_original) and on_boundary)
#################################################

T_hot = 360
T_cold = 300
T_ambient = 300
htc =200
omega = 0.1  # angular velocity

using_DG_solver = False
interactively = True
element_degree = 1

velocity_code = '''

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

bcs = { 
        "hot": {'boundary': bpin, 'boundary_id': 1, 'values': {
                    'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_hot)}
                 } },
        "cold": {'boundary': bdisc, 'boundary_id': 0, 'values': {
                    'temperature': {'variable': 'temperature', 'type': 'HTC', 'value': Constant(htc), 'ambient': Constant(T_ambient)}
                    #'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_hot-50)}
                 } },
}

settings = {'solver_name': 'ScalerEquationSolver',
                'mesh': mesh, 'periodic_boundary': None, 'fe_degree': element_degree,
                'boundary_conditions': bcs,
                'body_source': None,
                'initial_values': {'temperature': T_ambient},
                'material':{'density': 1000, 'specific_heat_capacity': 4200, 'thermal_conductivity':  1}, 
                'solver_settings': {
                    'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.1, 'ending_time': 1},
                    'reference_values': {'temperature': T_ambient},
                    'solver_parameters': {"relative_tolerance": 1e-9,  # mapping to solver.parameters of Fenics
                                                    "maximum_iterations": 500,
                                                    "monitor_convergence": True,  # print to console
                                                    },
                    },
                # solver specific settings
                'scaler_name': 'temperature',
                }

vector_degree = element_degree+1
v_e = Expression(cppcode=velocity_code, degree=vector_degree)
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
solver.convective_velocity = velocity

set_log_level(3)
T = solver.solve()
T = solver.plot()
if interactively:
    interactive()