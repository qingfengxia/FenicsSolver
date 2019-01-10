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
import math
import numpy as np

from dolfin import *
from FenicsSolver.ScalarTransportSolver  import ScalarTransportSolver

from config import is_interactive
interactively = is_interactive()

#mesh = UnitCubeMesh(20, 20, 20)
mesh = UnitSquareMesh(40, 40)
Q = FunctionSpace(mesh, "CG", 1)
#print(dir(Q))
#print(dir(Q._ufl_element))
#print(Q._ufl_element.degree())

cx_min, cy_min, cx_max, cy_max = 0,0,1,1
# no need for on_boundary, it should capture the boundary
top = AutoSubDomain(lambda x:  near(x[1], cy_max) )
bottom = AutoSubDomain(lambda x: near(x[1],cy_min))
left = AutoSubDomain(lambda x:  near(x[0], cx_min) )
right = AutoSubDomain(lambda x: near(x[0],cx_max))

T_hot = 360
T_cold = 300
T_ambient = 300

nonlinear = True
if nonlinear:
    conductivity = 0.6
else:
    conductivity = lambda T: (T-T_ambient)/T_ambient * 0.6
length = cy_max - cy_min
heat_flux = (T_hot-T_cold)/length*conductivity  # divided by length scale which is unity 1 ->  heat flux W/m^2

bcs = { 
        "hot": {'boundary': top, 'boundary_id': 1, 'values': {
                    'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_hot)}
                 } },
        "left":  {'boundary': left, 'boundary_id': 3, 'values': {
                 'temperature': {'variable': 'temperature', 'type': 'heatFlux', 'value': Constant(0)}
                 } },  # unit: K/m
        "right":  {'boundary': right, 'boundary_id': 4, 'values': {
                 #'temperature': {'variable': 'temperature', 'type': 'heatFlux', 'value': Constant(0)}
                 'temperature': {'variable': 'temperature', 'type': 'symmetry', 'value': None}
                 } }
        #back and front is zero gradient, need not set, it is default
}

settings = {'solver_name': 'ScalerEquationSolver',
                'mesh': None, 'function_space': Q, 'periodic_boundary': None, 'fe_degree': 1,
                'boundary_conditions': bcs,
                'body_source': None,
                'initial_values': {'temperature': T_ambient},
                'material':{'density': 1000, 'specific_heat_capacity': 4200, 'thermal_conductivity':  0.1}, 
                'solver_settings': {
                    'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.1, 'ending_time': 1},
                    'reference_values': {'temperature': T_ambient},
                    'solver_parameters': {"relative_tolerance": 1e-9,  # mapping to solver.parameters of Fenics
                                                    "maximum_iterations": 500,
                                                    "monitor_convergence": True,  # print to console
                                                    },
                    },
                # solver specific settings
                'scalar_name': 'temperature',
                }

K_anisotropic = Expression((('exp(x[0])','sin(x[1])'), ('sin(x[0])','tan(x[1])')), degree=0)  #works!
"""
# Create mesh functions for c00, c01, c11
c00 = MeshFunction("double", mesh, 2)  # degree/dim == 2?
c11 = MeshFunction("double", mesh, 2)
c22 = MeshFunction("double", mesh, 2)
for cell in cells(mesh):
    c00[cell] = conductivity
    c11[cell] = conductivity*0.1
    c22[cell] = conductivity*0.1
# Code for C++ evaluation of conductivity
conductivity_code = '''

class Conductivity : public Expression
{
public:

  // Create expression with 3 components
  Conductivity() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const uint D = cell.topological_dimension;
    const uint cell_index = cell.index;
    values[0] = (*c00)[cell_index];
    values[1] = (*c11)[cell_index];
    values[2] = (*c22)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<MeshFunction<double> > c00;
  std::shared_ptr<MeshFunction<double> > c11;
  std::shared_ptr<MeshFunction<double> > c22;

};
'''

c = Expression(cppcode=conductivity_code, degree=0)
c.c00 = c00
c.c11 = c11
c.c22 = c22
K = as_matrix(((c[0], c[1]), (c[1], c[2])))
"""

def setup(using_anisotropic_conductivity, using_convective_velocity, using_DG_solver, using_HTC):

    if using_anisotropic_conductivity:
        #tensor-weighted-poisson/python/demo_tensor-weighted-poisson.py
        K = K_anisotropic
    else:
        K = conductivity
        print("analytical heat flux [w/m^2] = ", heat_flux)

    if not using_HTC:
        if False: #using_convective_velocity:
            bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'values': {
                            'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_cold)}
                         } }
        else:
            bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'values': {
                            'temperature': {'variable': 'temperature', 'type': 'heatFlux', 'value': Constant(heat_flux)}
                         } }
    else:
        htc = 100
        bcs["hot"] = {'boundary': top, 'boundary_id': 1, 'values': {
                        'temperature': {'variable': 'temperature',  'type': 'heatFlux', 'value': Constant(heat_flux)}
                     } }
        bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'values': {
                        'temperature': {'variable': 'temperature', 'type': 'HTC', 'value': Constant(htc), 'ambient': Constant(T_ambient)}
                    } }

    if using_convective_velocity:
        settings['convective_velocity'] =  Constant((0.005, -0.005))
    else:
        settings['convective_velocity'] = None

    solver = ScalarTransportSolver(settings)

    solver.material['conductivity'] = K
    #debugging: show boundary selection
    plot(solver.boundary_facets, title="boundary facets colored by ID")  # matplotlib can not plot 1D
    plot(solver.subdomains, title="subdomain cells colored by ID")

    T = solver.solve()
    post_process(T, interactively)

def post_process(T, interactively):
    # Report flux, they should match
    normal = FacetNormal(mesh)
    boundary_facets = FacetFunction('size_t', mesh)
    boundary_facets.set_all(0)
    id=1
    bottom.mark(boundary_facets, id)
    ds= Measure("ds", subdomain_data=boundary_facets)

    flux = assemble(conductivity * dot(grad(T), normal)*ds(id))
    print("heat flux rate integral on the surface(w/m^2)", flux)

    plot(T, title='Temperature')
    if interactively:
        interactive()

def test_radiation():
    using_anisotropic_conductivity = False
    if using_anisotropic_conductivity:
        #tensor-weighted-poisson/python/demo_tensor-weighted-poisson.py
        K = K_anisotropic
    else:
        K = conductivity
        htc = heat_flux / (T_hot - T_cold)
        print("analytical heat flux [w/m^2] = ", heat_flux)

    #if using_convective_velocity:
    bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'values': {
                    'temperature': {'variable': 'temperature', 'type': 'Dirichlet', 'value': Constant(T_cold)}
                 } }
    settings['radiation_settings'] = {'ambient_temperature': T_ambient-20, 'emissivity': 0.9}
    settings['convective_velocity'] = None
    solver = ScalarTransportSolver(settings)
    solver.material['conductivity'] = K
    solver.material['emissivity'] = 0.9

    T = solver.solve()
    post_process(T, interactively)

def test():
    #setup(using_anisotropic_conductivity = True, using_convective_velocity = False, using_DG_solver = False, using_HTC = False)
    #setup(using_anisotropic_conductivity = False, using_convective_velocity = False, using_DG_solver = False, using_HTC = True)
    #DG is not test here
    setup(using_anisotropic_conductivity = False, using_convective_velocity = True, using_DG_solver = True, using_HTC = True)

if __name__ == '__main__':
    test()
    test_radiation()