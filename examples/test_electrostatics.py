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

from FenicsSolver.ScalerTransportSolver import ScalerTransportSolver

#mesh = UnitCubeMesh(20, 20, 20)
mesh = UnitSquareMesh(40, 40)
Q = FunctionSpace(mesh, "CG", 1)

cx_min, cy_min, cx_max, cy_max = 0,0,1,1
# no need for on_boundary, it should capture the boundary
top = AutoSubDomain(lambda x:  near(x[1], cy_max) )
bottom = AutoSubDomain(lambda x: near(x[1],cy_min))
left = AutoSubDomain(lambda x:  near(x[0], cx_min) )
right = AutoSubDomain(lambda x: near(x[0],cx_max))

V_high = 360
V_low = 300
V_ground = 300
resistivity = 2300 # ohm.m
# source item: unit:  C/m3
# flux boundary condition (surface charge):    C/m2
# Neumann boundary, unit  V/m
# PointSource  (Dirac delta function) is supported,  uint?   
# API: PointSource(V, p, magnitude=1.0),  Create point source at given coordinates point of given magnitude


#polarity is not considered by this solver

magnetic_permeability_0 = 4 * pi * 1e-7
electric_permittivity_0 = 8.854187817e-12
K_anisotropic = Expression((('exp(x[0])','sin(x[1])'), ('sin(x[0])','tan(x[1])')), degree=0)

# dielectric or conducting
material = {'name': "silicon", 'thermal_conductivity': 149, 'specific_heat_capacity': 1000, \
                    'density': 2500, 'sound_speed': 8433, 'Poisson_ratio': 0.2, 'elastic_modulus': 1.5e11, \
                    'magnetic_permeability': magnetic_permeability_0*1, \
                    'relative_electric_permittivity': 11.7, 'electric_conductivity': 1.0/resistivity}  # Tref set in reference_values


length = cy_max - cy_min
electric_displacement = (V_high-V_low)/length * material['electric_permittivity']   # divided by length scale which is unity 1 ->  heat flux W/m^2

bcs = { 
        "hot": {'boundary': top, 'boundary_id': 1, 'type': 'Dirichlet', 'value': Constant(V_high)}, 
        "left":  {'boundary': left, 'boundary_id': 3, 'type': 'flux', 'value': Constant(0)},
        "right":  {'boundary': right, 'boundary_id': 4, 'type': 'flux', 'value': Constant(0)}, 
        #back and front is zero gradient, need not set, it is default
}

settings = {'solver_name': 'ScalerEquationSolver',
                'mesh': None, 'function_space': Q, 'periodic_boundary': None, 'element_degree': 1,
                'boundary_conditions': bcs, 'body_source': None, 
                'initial_values': {'electric_potential': V_ground},
                'material': material, 
                'solver_settings': {
                    'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.1, 'ending_time': 1},
                    'reference_values': {'temperature': 300, 'electric_potential': V_ground},
                    'solver_parameters': {"relative_tolerance": 1e-9,  # mapping to solver.parameters of Fenics
                                                    "maximum_iterations": 500,
                                                    "monitor_convergence": True,  # print to console
                                                    },
                    },
                # solver specific settings
                'scaler_name': 'electric_potential',
                }

def test(interactively = False):
    using_convective_velocity = False

    using_anisotropic_material = True
    if using_anisotropic_material:
        #tensor-weighted-poisson/python/demo_tensor-weighted-poisson.py
        settings['material']['electric_permittivity'] = K_anisotropic
    else:
        print("analytical current density [A/m^2] = ", electric_displacement)

    #if using_convective_velocity:
    bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'type': 'Dirichlet', 'value': Constant(V_low)}
    #bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'type': 'flux', 'value': Constant(electric_displacement)}

    if using_convective_velocity:
        settings['convective_velocity'] =  Constant((0.5, -0.5))
    else:
        settings['convective_velocity'] = None
    solver = ScalerTransportSolver(settings)
    #debugging: show boundary selection
    plot(solver.boundary_facets, "boundary facets colored by ID")
    plot(solver.subdomains, "subdomain cells colored by ID")

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

    flux = assemble(settings['material']['electric_permittivity'] * dot(grad(T), normal)*ds(id))
    print("integral on the top surface(A)", flux)

    plot(T, title='electric Potential (V)')
    #plot(mesh)
    if interactively:
        interactive()

if __name__ == '__main__':
    test()
