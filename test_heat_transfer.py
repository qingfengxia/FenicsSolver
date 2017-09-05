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

from ScalerEquationSolver import ScalerEquationSolver

def test(using_anisotropic_conductivity = True):
    #mesh = UnitCubeMesh(20, 20, 20)
    mesh = UnitSquareMesh(20, 20)
    Q = FunctionSpace(mesh, "CG", 1)

    cx_min, cy_min, cx_max, cy_max = 0,0,1,1
    # no need for on_boundary, it should capture the boundary
    top = AutoSubDomain(lambda x:  near(x[1], cy_max) )
    bottom = AutoSubDomain(lambda x: near(x[1],cy_min))
    left = AutoSubDomain(lambda x:  near(x[0], cx_min) )
    right = AutoSubDomain(lambda x: near(x[0],cx_max))

    T_hot = 60
    T_cold = 30
    T_ambient = 0
    conductivity = 0.6
    heat_flux_density = (T_hot-T_cold)*conductivity

    bcs = { 
            "hot": {'boundary': top, 'boundary_id': 1, 'type': 'Dirichlet', 'value': Constant(T_hot)}, 
            "left":  {'boundary': left, 'boundary_id': 3, 'type': 'Neumann', 'value': Constant(0)}, 
            "right":  {'boundary': right, 'boundary_id': 4, 'type': 'Neumann', 'value': Constant(0)}, 
            #back and front is zero gradient, need not set, it is default
    }

    if using_anisotropic_conductivity:
        #tensor-weighted-poisson/python/demo_tensor-weighted-poisson.py
        K = Expression((('exp(x[0])','sin(x[1])'), ('sin(x[0])','tan(x[1])')), degree=0)  #works!
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
    else:
        K = conductivity
        htc = heat_flux_density / (T_cold - T_ambient)
        print("analytical heat flux density = ", heat_flux_density)

    #bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'type': 'Dirichlet', 'value': Constant(T_cold)}
    bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'type': 'Neumann', 'value': Constant(heat_flux_density)}
    #bcs["cold"] = {'boundary': bottom, 'boundary_id': 2, 'type': 'Robin', 'value': (Constant(htc), Constant(T_ambient))}

    settings = {'solver_name': 'ScalerEquationSolver',
                    'mesh': None, 'function_space': Q, 'periodic_boundary': None, 
                    'boundary_conditions': bcs, 'body_source': None, 
                    'initial_values': {'temperature': 300},
                    'material':{}, 
                    'solver_settings': {
                        'transient_settings': {'transient': False, 'starting_time': 0, 'time_step': 0.01, 'ending_time': 0.03},
                        'reference_values': {'temperature': 300},
                        'convergence_settings': {'default': 1e-3}
                        },
                    # solver specific settings
                    'scaler_name': 'temperature',
                    "convective_velocity": Constant((0.1, 0))
                    }

    solver = ScalerEquationSolver(settings)
    solver.material['conductivity'] = K
    T = solver.solve()

    # Report flux, they should match
    normal = FacetNormal(mesh)
    boundary_facets = FacetFunction('size_t', mesh)
    boundary_facets.set_all(0)
    id=1
    bottom.mark(boundary_facets, id)
    ds= Measure("ds", subdomain_data=boundary_facets)
    if not using_anisotropic_conductivity:
        flux = assemble(K*dot(grad(T), normal)*ds(id))
        print("tuft heat flux rate integral on the surface(w/m^2)", flux)

    plot(T, title='Temperature')
    #plot(mesh)
    interactive()

if __name__ == '__main__':
    test()