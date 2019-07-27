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

from config import is_interactive
interactively = is_interactive()

"""
This is an test program to test various convection scheme, it does not user FenicsSolver
"""

def defined(x):
    return x in locals() or x in globals()

#info(parameters,True)  # list all parameters
#parameters['plotting_backend'] = 'matplotlib'
#parameters['std_out_all_processes'] = 0
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

#######################################################
has_convection = True
using_DG_solver = False
fe_degree = 1 # 2 is possible but use more memory
vector_degree = fe_degree

if using_DG_solver:
    heat_source = None
else:
    heat_source = 1e5

set_log_level(ERROR)

fe_degree = 1 # 2 is possible but use more memory
vector_degree = fe_degree  +1


def defined(x):
    return x in locals() or x in globals()

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

#######################################################
r_d = 0.5
omega = 50  # angular velocity 0.005 is correct without compensation, 

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 40, 40, 'crossed')

#ic= Expression("((pow(x[0]-0.25,2)+pow(x[1]-0.25,2))<0.2*0.2)?(-25*((pow(x[0]-0.25,2)+pow(x[1]-0.25,2))-0.2*0.2)):(0.0)")
#ic= Expression("((pow(x[0]-0.3,2)+pow(x[1]-0.3,2))<0.2*0.2)?(1.0):(0.0)", domain=mesh)

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
    double r_r= 1 - ((r-r_d*0.5)/(r_d*0.5))*((r-r_d*0.5)/(r_d*0.5));
    double v = omega * r_r * r_d;
      
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
'''%(0.5, 0.5, omega, r_d)

#v_e = Expression(cppcode=velocity_code_2D, degree=vector_degree)  # SPUG does not work for rotating speed, IP works
v_e = Expression(("-(x[1]-0.5)","(x[0]-0.5)"), degree=vector_degree)  # works for both IP and SPUG
if has_convection:
    W = VectorFunctionSpace(mesh, 'CG', vector_degree)
    velocity = interpolate(v_e, W)
    plot(velocity, title="velocity")  # matplotlib 

T_hot = 360
T_cold = 300
T_ambient = 300
htc = 100

boundary_id_pin = 2
boundary_id_disc = 1

bpin = AutoSubDomain(lambda x, on_boundary:  near(x[1], 1) and on_boundary)
bdisc = AutoSubDomain(lambda x, on_boundary: near(x[1], 0) and on_boundary)

bcs = { 
        "hot": {'boundary': bpin, 'boundary_id': boundary_id_pin, 'values': {
                    #'temperature': {'variable': 'temperature', 'type': 'HTC', 'value': Constant(htc), 'ambient': Constant(T_ambient)}
                    #'temperature': {'variable': 'temperature', 'type': 'Neumann', 'value':0}
                    'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_hot)}
                } },
        "cold": {'boundary': bdisc, 'boundary_id': boundary_id_disc, 'values': {
                    #'temperature': {'variable': 'temperature', 'type': 'HTC', 'value': Constant(htc), 'ambient': Constant(T_ambient)}
                    'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_cold)}
                } },
}

settings = {'solver_name': 'ScalarTransportSolver',
            'mesh': mesh, 'periodic_boundary': None, 'fe_degree': fe_degree,
            'boundary_conditions': bcs,
            'body_source': heat_source,  # there must be a source for DG solver
            'convective_velocity': velocity,
            'initial_values': {'temperature': T_ambient},
            'material':{'density': 1000, 'specific_heat_capacity': 4200, 'thermal_conductivity': 15}, 
            'solver_settings': {
                'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.1, 'ending_time': 2},
                'reference_values': {'temperature': T_ambient},
                'solver_parameters': {"relative_tolerance": 1e-11,  # mapping to solver.parameters of Fenics
                                                "maximum_iterations": 500,
                                                "monitor_convergence": True,  # print to console
                                                },
                },
            # solver specific settings
            'scalar_name': 'temperature',
            }


if using_DG_solver:
    # DG solver still FAIL !!!!
    from FenicsSolver.ScalerTransportDGSolver import ScalerTransportDGSolver
    solver = ScalerTransportDGSolver(settings)
    #velocity = interpolate(v_e, solver.vector_function_space)
else:
    #settings['advection_settings'] = {'stabilization_method': 'SPUG', 'Pe': 1.0/(15.0/(4200*1000))}
    # SPUG method 1 has on effect, compared with no stabilization_method???
    # it is possible to use SPUG (with rotating velocity),  with and without body source
    # with body source, SPUG and IP give different result temperature field!!!
    #settings['advection_settings'] = {'stabilization_method': 'IP', 'alpha': 0.1}
    #result can be sensitive to alpha value!!!
    #without any stabilization_method, solver will not get correct answer
    from FenicsSolver.ScalarTransportSolver import ScalarTransportSolver
    solver = ScalarTransportSolver(settings)


plot(mesh, title="mesh")
#plot(solver.boundary_facets, title="boundary facets colored by ID")  # matplotlib can not plot 1D

#DG for conductivity and heat source
#Q = solver.function_space
#DG0 = FunctionSpace(solver.mesh, 'DG', 0)  # can not mixed higher order CG and DG?
#solver.body_source = interpolate(s_e, DG0)
#plot(solver.body_source )

T = solver.solve()

ds= Measure("ds", subdomain_data=solver.boundary_facets)
cooling_disc = assemble(htc*(T-T_ambient)*ds(boundary_id_disc))
print('convective cooling from htc boundary is : ', cooling_disc)

if interactively:
    solver.plot()