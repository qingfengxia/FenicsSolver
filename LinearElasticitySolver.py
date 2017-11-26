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
import collections
import numbers
import numpy as np


#####################################
from dolfin import *
from mshr import Box

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"


from SolverBase import SolverBase, SolverError
class LinearElasticitySolver(SolverBase):
    """ transient and thermal stress is implemented but not tested
    contact boundary is not implemented,
    placity (nonlinear elastic) will be implemented in another solver
    """
    def __init__(self, case_settings):
        SolverBase.__init__(self, case_settings)

        self.settings['vector_name'] = 'displacement'
        # there must be a value for body force as source item, to make L not empyt in a == L
        #todo: moved to SolverBase
        if self.body_source:
            self.body_source = self.translate_value(self.body_source)
        else:
            if self.dimension == 3:
                self.body_source = Constant((0, 0, 0))
            else:
                self.body_source = Constant((0, 0))

        # thermal stress, material 
        if 'temperature_distribution' in case_settings:
            self.thermal_stress = True
            self.temperature_distribution = case_settings['temperature_distribution']

        self.solving_modal = False

    def set_function_space(self, mesh_or_function_space, periodic_boundary):
        try:
            self.mesh = mesh_or_function_space
            if periodic_boundary:
                self.function_space = VectorFunctionSpace(self.mesh, "CG", self.degree, constrained_domain=periodic_boundary)
                # the group and degree of the FE element.
            else:
                self.function_space = VectorFunctionSpace(self.mesh, "CG", self.degree)
        except:
            self.function_space = mesh_or_function_space
            self.mesh = self.function_space.mesh()
        self.is_mixed_function_space = False  # how to detect it is mixed, vector, scaler , tensor?


    def sigma(self, u):
        # Stress computation
        elasticity = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = elasticity/(2.0*(1.0 + nu))
        lmbda = elasticity*nu/((1.0 + nu)*(1.0 - 2.0*nu))
        return 2.0*mu*sym(grad(u)) + lmbda*tr(sym(grad(u)))*Identity(len(u))

    def von_Mises(self, u):
        s = self.sigma(u) - (1./3)*tr(self.sigma(u))*Identity(self.dimension)  # deviatoric stress
        von_Mises = sqrt(3./2*inner(s, s))
        
        V = FunctionSpace(self.mesh, 'P', 1)  # correct, but why using another function space
        return project(von_Mises, V)

    def strain_energy(self, u):
        # Strain energy or the plastic heat generation
        elasticity = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = elasticity/(2.0*(1.0 + nu))
        lmbda = elasticity*nu/((1.0 + nu)*(1.0 - 2.0*nu))
        return lmbda/2.0*(tr(eps(v)))^2 + mu*tr(eps(v)**2)

    def update_boundary_conditions(self, time_iter_, u, v, ds):
        V = self.function_space
        bcs = []
        integrals_F = []
        mesh_normal = FacetNormal(self.mesh)  # n is predefined as normal?
        for name, bc_settings in self.boundary_conditions.items():
            i = bc_settings['boundary_id']
            bc = self.get_boundary_variable(bc_settings)
            print(bc)
            if bc['type'] =='Dirichlet' or bc['type'] =='displacement':
                bv = self.translate_value(bc['value'])
                if isinstance(bv, (tuple, list)) and len(bv) == self.dimension:
                    axis_i=0
                    for disp in bv:
                        if not disp is None:  # None means free of constraint, but zero is kind of constraint
                            dbc = DirichletBC(V.sub(axis_i), self.translate_value(disp), self.boundary_facets, i)
                            bcs.append(dbc)
                        axis_i += 1
                else:
                    dbc = DirichletBC(V, self.translate_value(bv), self.boundary_facets, i)
                    bcs.append(dbc)
            elif bc['type'] == 'force':
                bc_force = self.translate_value(bc['value'])
                # calc the surface area and calc stress, normal and tangential?
                bc_area = assemble(Constant(1)*ds(bc['boundary_id'], domain=self.mesh))
                print('boundary area (m2) for force boundary is', bc_area)
                g = bc_force / bc_area
                # FIXME: assuming all force are normal to mesh boundary
                if 'direction' in bc and bc['direction']:
                    direction_vector = bc['direction']
                else:
                    direction_vector = mesh_normal
                integrals_F.append( dot(g,v)*ds(i))
            elif bc['type'] == 'stress' or bc['type'] =='Neumann':
                if 'direction' in bc and bc['direction']:
                    direction_vector = bc['direction']
                else:
                    direction_vector = mesh_normal  # normal to boundary surface, n is predefined
                g = self.translate_value(bc['value'])
                #FIXME: assuming all force are normal to mesh boundary
                integrals_F.append(dot(g,v)*ds(i))
            elif bc['type'] == 'symmetry':
                raise SolverError('symmetry boundary type`{}` is not supported'.format(bc['type']))
            else:
                raise SolverError('boundary type`{}` is not supported'.format(bc['type']))
        ## nodal constraint is not yet supported, try make it a small surface load instead
        return bcs, integrals_F

    def generate_form(self, time_iter_, u, v, u_current, u_prev):
        V = self.function_space
        # Define variational problem
        #u = TrialFunction(V)
        #v = TestFunction(V)

        elasticity = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        mu = elasticity/(2.0*(1.0 + nu))
        lmbda = elasticity*nu/((1.0 + nu)*(1.0 - 2.0*nu))

        F = inner(self.sigma(u), grad(v))*dx

        ds= Measure("ds", subdomain_data=self.boundary_facets)  # if later marking updating in this ds?
        bcs, integrals_F = self.update_boundary_conditions(time_iter_, u, v, ds)
        if time_iter_==0:
            plot(self.boundary_facets, title = "boundary facets colored by ID")

        if self.body_source:
            integrals_F.append( inner(self.body_source, v)*dx )

        ## thermal stress not tested, todo: thermal_stress_settings = {}
        if self.temperature_distribution:
            T = self.translate_value(self.temperature_distribution)  # interpolate
            tec = self.material['thermal_expansion_coefficient']
            # only apply if the body is NOT freely expensible in all directions, with displacement constraint
            # if there is one or two directions can have free expansion, poisson_ratio should be considerred
            thermal_strain = tec * ( T - Constant(self.reference_values['temperature']))
            if self.dimension == 3:
                thermal_stress = inner(elasticity * thermal_strain , v)*dx
            elif self.dimension == 2:  # assuming free expansion the third direction
                thermal_strain = thermal_strain * (1 - self.material['poisson_ratio'])
                thermal_stress = inner(elasticity * thermal_strain , v)*dx
            else:
                raise SolverError('only 3D and 2D simulation is supported')
            # there is another case: local hotspot but not melt
            integrals_F.append( thermal_stress )

        # Assemble system, applying boundary conditions and extra items
        if len(integrals_F):
            for item in integrals_F: F += item  # L side

        return F, bcs

    def solve_static(self, F, u_, bcs):
        #if self.is_iterative_solver:
        #u_ = self.solve_iteratively(F, bcs, u)
        u_ = self.solve_amg(F, u_, bcs)
        # calc boundingbox to make sure no large deformation?
        return u_

    def solve(self):
        u = self.solve_transient()  # defined in SolverBase
        
        if self. solving_modal:
            self.solve_modal(F, bcs)  # test passed

        return u

    def solve_modal(self, F, bcs):
        # todo: Assemble stiffness form, it is not fully tested yet
        A = PETScMatrix()
        b = PETScVector()
        '''
        assemble(a, tensor=A)
        for bc in bcs:
            bc.apply(A)          # apply the boundary conditions
        '''
        assemble_system(lhs(F), rhs(F), bcs, A_tensor=A, b_tensor=b)  # preserve symmetry

        # Create eigensolver
        eigensolver = SLEPcEigenSolver(A)

        # Compute all eigenvalues of A x = \lambda x
        print("Computing eigenvalues. This can take a minute.")
        eigensolver.solve()

        # Extract largest (first) eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(0)

        print("Largest eigenvalue: ", r)

        # Initialize function and assign eigenvector
        ev = Function(self.function_space)
        ev.vector()[:] = rx

        return ev


if __name__ == '__main__':
    test()