#Time-integration of elastodynamics equation
# it should be marged into transient of elastic solver.  damping
#https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
    def generate_function_space(self, periodic_boundary):
        self.vel_degree = self.settings['fe_degree'] + 1  # order 3 is working for 2D elbow testing
        self.pressure_degree = self.settings['fe_degree']
        self.is_mixed_function_space = True  # FIXME: how to detect it is mixed, if function_space is provided

        V = VectorElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.vel_degree)  # degree 2, must be higher than pressure
        Q = FiniteElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.pressure_degree)
        #T = FiniteElement("CG", self.mesh.ufl_cell(), 1)  # temperature subspace, or just use Q
        if self.solving_temperature:
            mixed_element = MixedElement([V, V, Q])
        else:
            mixed_element = MixedElement([V, V, Q, Q])
        if periodic_boundary:
            self.function_space = FunctionSpace(self.mesh, mixed_element, constrained_domain=periodic_boundary)
        else:
            self.function_space = FunctionSpace(self.mesh, mixed_element)