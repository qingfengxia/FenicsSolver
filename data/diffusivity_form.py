    def generate_form(self, time_iter_, T, Tq, T_current, T_prev):
        # T, Tq can be shared between time steps, form is unified diffussion coefficient
        normal = FacetNormal(self.mesh)

        dx= Measure("dx", subdomain_data=self.subdomains)  # 
        ds= Measure("ds", subdomain_data=self.boundary_facets)

        k = self.conductivity() # constant, experssion or tensor
        capacity = self.capacity()  # density * specific capacity -> volumetrical capacity
        diffusivity = self.diffusivity()  # diffusivity

        bcs, integrals_N = self.update_boundary_conditions(time_iter_, T, Tq, ds)
        # boundary type is defined in FreeCAD FemConstraintFluidBoundary and its TaskPanel
        # zeroGradient is default thermal boundary, no effect on equation?

        def get_source_item():
            if isinstance(self.body_source, dict):
                S = []
                for k,v in self.get_body_source().items():
                    # it is good to using DG for multi-scale meshing, subdomain marking double
                    S.append(v['value']*Tq*dx(v['subdomain_id']))
                return sum(S)
            else:
                if self.body_source:
                    return  self.get_body_source()*Tq*dx
                else:
                    return None

        # poission equation, unified for all kind of variables
        def F_static(T, Tq):
            F =  inner( diffusivity * grad(T), grad(Tq))*dx
            F -= sum(integrals_N) * Constant(1.0/capacity)
            return F

        def F_convective():
            h = CellSize(self.mesh)  # cell size
            velocity = self.get_convective_velocity_function(self.convective_velocity)
            if self.transient_settings['transient']:
                dt = self.get_time_step(time_iter_)
                # Mid-point solution
                T_mid = 0.5*(T_prev + T)
                # Residual
                res = (T - T_prev)/dt + (dot(velocity, grad(T_mid)) -  diffusivity * div(grad(T_mid)))  # does not support conductivity tensor
                # Galerkin variational problem
                F = Tq*(T - T_prev)/dt*dx + (Tq*dot(velocity, grad(T_mid))*dx +  diffusivity * dot(grad(Tq), grad(T_mid))*dx)
            else:
                T_mid = T
                # Residual
                res = dot(velocity, grad(T_mid)) -  diffusivity * div(grad(T_mid))
                #print(res)
                # Galerkin variational problem
                F = Tq*dot(velocity, grad(T_mid))*dx + \
                     diffusivity * dot(grad(Tq), grad(T_mid))*dx

            F -= sum(integrals_N) * Constant(1.0/capacity)  # included in F_static()
            if self.body_source:
                res -= get_source_item() * Constant(1.0/capacity)
                F -= get_source_item() * Constant(1.0/capacity)*Tq*dx  # check
            # Add SUPG stabilisation terms
            vnorm = sqrt(dot(velocity, velocity))
            F += (h/(2.0*vnorm))*dot(velocity, grad(Tq))*res*dx
            return F

        if self.convective_velocity:  # convective heat conduction
            F = F_convective()
        else:
            if self.transient_settings['transient']:
                dt = self.get_time_step(time_iter_)
                theta = Constant(0.5) # Crank-Nicolson time scheme
                # Define time discretized equation, it depends on scaler type:  Energy, Species,
                F = (1.0/dt)*inner(T-T_prev, Tq)*dx \
                       + theta*F_static(T, Tq) + (1.0-theta)*F_static(T_prev, Tq)  # FIXME:  check using T_0 or T_prev ? 
            else:
                F = F_static(T, Tq)
            #print(F, get_source_item())
            if self.body_source:
                F -= get_source_item() * Constant(1.0/capacity)

        if self.scaler_name == "temperature" and self.has_radiation:
            Stefan_constant = 5.670367e-8  # W/m-2/K-4
            if 'emissivity' in self.material:
                emissivity = self.material['emissivity']  # self.settings['radiation_settings']['emissivity'] 
            else:
                emissivity = 1.0
            T_ambient_radiaton = self.radiation_settings['ambient_temperature']
            m_ = emissivity * Stefan_constant
            radiation_flux = m_*(pow(T, 4) - T_ambient_radiaton**4)  # it is nonlinear item
            print(m_, radiation_flux, F)
            F -= radiation_flux*Tq*ds * Constant(1.0/capacity)  # for all surface, without considering view angle
            F = action(F, T_current)  # API 1.0 still working ; newer API , replacing TrialFunction with Function for nonlinear 
            self.J = derivative(F, T_current, T)  # Gateaux derivative
        return F, bcs
