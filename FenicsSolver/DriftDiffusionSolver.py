from dolfin import *
from dolfin.cpp.la import NewtonSolver

from .SolverBase import SolverBase, SolverError
from .ScalarTransportSolver import ScalarTransportSolver



class DriftDiffusionSolver(SolverBase):
    """  a drift-diffusion transportation solver, exampled by simiconductor electron and cavity
    boundary condition can not be shared for scalar transportation due to mixed funciton space
    """

    def __init__(self, s):
        SolverBase.__init__(self, s)
        
    def generate_function_space(self, periodic_boundary):
        self.is_mixed_function_space = True

        V = VectorElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.settings['fe_degree'] +1) 
        Q = FiniteElement(self.settings['fe_family'], self.mesh.ufl_cell(), self.settings['fe_degree'])
        #if temperature is also needed to solve, then one more scalar
        # for plasma, the mass might be compressible, also there is convection
        mixed_element = MixedElement([Q, Q, Q])

        if periodic_boundary:
            self.function_space = FunctionSpace(self.mesh, mixed_element, constrained_domain=periodic_boundary)
        else:
            self.function_space = FunctionSpace(self.mesh, mixed_element)

    def generate_form(self, time_iter_, du_mixed, v_mixed, u_current, u_prev):
        #http://www.iue.tuwien.ac.at/phd/triebl/node15.html

        u, un, up = split(u_current)  # u: equilibrium potential (intrinsic energy), e and hole energy
        u_prev, un_prev, up_prev = split(u_prev) 
        v, vn, vp = split(v_mixed)
        du, dn, dp = split(du_mixed)

        k_B = 1.38064852e-23 # Boltzmann Coeff, unit: J/K
        epsilon = Constant(1.)  # eletric permitivity
        T = Constant(1.)  # lattice temperature, input field
        ni = Ni(degree = 1) # Carrier concentration, Intrinsic concentration 
        q = 1.0  # charge per e, hole/donator, acceptor
        
        d = Doping(degree = 1)  #input, hole distribution field
        n = ni*exp((u-un)*q/(k_B*T))  # Carrier density equations
        p = ni*exp((up-u)*q/(k_B*T))  # ERROR? k_B is missing from example

        rho = (p+d-n)*q  # Charge density , acceptor is not included here, 
        D_n = Constant(100.0)  # diffusivity
        D_p = Constant(100.0)

        #J_p/q = p*mobility_p*E - D*mu_p*grad(p) - p*mu_p_T*grad(T)
        # D_p = mobility_p * (k_B*T)/q
        # also thermal diffusion due to lattice temmperature gradient
        tau_n = Constant(1.0)  # ? 
        tau_p = Constant(1.0)
        R = (n*p - ni**2) + (n*p-ni**2)/(tau_n*(p+ni**2)+tau_p*(n+ni**2))  # recombination rate

        #dx = Measure("dx")[cell_domains]
        #dS = Measure("dS")

        L1 = inner(epsilon*grad(u),grad(v))*dx - rho*v*dx  # current/q, 
        L2 = inner(D_n*n*grad(un), grad(vn))*dx - R*vn*dx #  div(J_n/q) - transient = R
        L3 = inner(D_p*p*grad(up), grad(vp))*dx + R*vp*dx #  div(J_p/q) + transient = -R
        # transient item:  q*(un-un_prev)/dt
        # continuity equations of n and p should take into combination and regeneration as source
        #dn/dt - div(J_n)/q = S_n
        #dp/dt + div(J_p)/q = S_p

        F = L1+L2+L3
        J = derivative(F,u_current,du_mixed)

    def solve_form(self, F, u_, bcs):
        #
        return u_
        

class DriftDiffusionSolver(NewtonSolver):

  def __init__(self, du_max=1.1):
    self.du_max = du_max
    NewtonSolver.__init__(self)

  def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem, interation):
    self.damp_potentials(x, dx)
    #NewtonSolver.update_solution(self, x, dx)  # outdated API
    #new cpp api:  update_solution(x, dx, relaxation_parameter, nonlinear_problem, interation)
    NewtonSolver.update_solution(self, x, dx, relaxation_parameter, nonlinear_problem, interation)

  def damp_potentials(self, x, dx):
    """This will strongly influence the convergence"""

    du = dx.array()
    dumax = self.du_max

    i = (du > dumax)
    du[i] = dumax#*(1 + np.log(du[i])/dumax)
    i = (du < -dumax)
    du[i] = -dumax#*(1 + np.log(-du[i])/dumax)
    dx[:] = du

class DriftDiffusionProblem(NonlinearProblem):
  def __init__(self, F, J, bcs=None, form_compiler_parameters=None, ident_zeros=False):
    NonlinearProblem.__init__(self)
    self._F = F
    self._J = J
    self.bcs = bcs
    self.form_compiler_parameters = form_compiler_parameters
    self.ident_zeros = ident_zeros

  def F(self, b, x):
    assemble(self._F, tensor=b, form_compiler_parameters=self.form_compiler_parameters)

    #Apply boundary conditions
    for bc in self.bcs:
        bc.apply(b, x)

  def J(self, A, x):
    assemble(self._J, tensor=A,
             form_compiler_parameters=self.form_compiler_parameters)

    #Apply boundary conditions
    for bc in self.bcs:
        bc.apply(A)

    if self.ident_zeros:
        A.ident_zeros()