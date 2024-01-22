import firedrake as fd
import warpmesh as wm
import matplotlib.pyplot as plt
import numpy as np


class SwirlEvaluator():
    """
    Evaluate error for advection swirl problem:
        1. Solver implementation for the swirl problem
        2. Error & Time evaluation
    """
    def __init__(self, mesh, mesh_fine, mesh_new, dataset, **kwargs):
        """
        Init the problem:
            1. define problem on fine mesh and coarse mesh
            2. init function space on fine & coarse mesh
        """
        self.mesh = mesh                    # coarse mesh
        self.mesh_fine = mesh_fine            # fine mesh
        self.mesh_new = mesh_new            # adapted mesh
        self.dataset = dataset              # dataset containing all data
        self.save_interval = kwargs.pop("save_interval", 5)

        # Init coords setup
        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = self.mesh_fine.coordinates.vector().array().reshape(-1, 2) # noqa
        self.best_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.adapt_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        # error measuring vars
        self.error_adapt_list = []
        self.error_og_list = []
        self.best_error_iter = 0

        # X and Y coordinates
        self.x, self.y = fd.SpatialCoordinate(mesh)
        self.x_fine, self.y_fine = fd.SpatialCoordinate(self.mesh_fine)

        # function space on coarse mesh
        self.scalar_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.vector_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        self.tensor_space = fd.TensorFunctionSpace(self.mesh, "CG", 1)
        # function space on fine mesh
        self.scalar_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        self.vector_space_fine = fd.VectorFunctionSpace(self.mesh_fine, "CG", 1)  # noqa

        # Test/Trial function on coarse mesh
        self.du_trial = fd.TrialFunction(self.scalar_space)
        self.phi = fd.TestFunction(self.scalar_space)
        # Test/Trial function on fine mesh
        self.du_trial_fine = fd.TrialFunction(self.scalar_space_fine)
        self.phi_fine = fd.TestFunction(self.scalar_space_fine)
        # normal function on coarse / fine mesh
        self.n = fd.FacetNormal(self.mesh)
        self.n_fine = fd.FacetNormal(self.mesh_fine)

        # simulation params
        self.T = kwargs.pop("T", 1)
        self.t = 0.0
        self.n_step = kwargs.pop("n_step", 500)
        self.threshold = self.T / 2                          # Time point the swirl direction get reverted  # noqa
        self.dt = self.T / self.n_step
        self.dtc = fd.Constant(self.dt)
        # initial condition params
        self.sigma = kwargs.pop("sigma", (0.05/6))
        self.alpha = kwargs.pop("alpha", 1.5)
        self.r_0 = kwargs.pop("r_0", 0.2)
        self.x_0 = kwargs.pop("x_0", 0.25)
        self.y_0 = kwargs.pop("y_0", 0.25)

        # initital condition of u on coarse / fine mesh
        u_init_exp = wm.get_u_0(self.x, self.y, self.r_0, self.x_0, self.y_0, self.sigma)  # noqa
        u_init_exp_fine = wm.get_u_0(self.x_fine, self.y_fine, self.r_0, self.x_0, self.y_0, self.sigma)  # noqa
        self.u_init = fd.Function(self.scalar_space).interpolate(u_init_exp)
        self.u_init_fine = fd.Function(self.scalar_space_fine).interpolate(u_init_exp_fine) # noqa
        # PDE vars on coarse & fine mesh
        #       solution field u
        self.u = fd.Function(self.scalar_space).assign(self.u_init)
        self.u1 = fd.Function(self.scalar_space)
        self.u2 = fd.Function(self.scalar_space)
        self.u_fine = fd.Function(self.scalar_space_fine).assign(self.u_init_fine)  # noqa
        self.u1_fine = fd.Function(self.scalar_space_fine)
        self.u2_fine = fd.Function(self.scalar_space_fine)
        self.u_in = fd.Constant(0.0)
        self.u_in_fine = fd.Constant(0.0)
        #       temp vars for saving u on coarse & fine mesh
        self.u_cur = fd.Function(self.scalar_space)             # solution from current time step  # noqa
        self.u_cur_fine = fd.Function(self.scalar_space_fine)
        self.u_hess = fd.Function(self.scalar_space)            # buffer for hessian solver usage  # noqa
        #       buffers
        self.u_fine_buffer = fd.Function(self.scalar_space_fine).assign(self.u_init_fine)  # noqa
        self.coarse_adapt = fd.Function(self.scalar_space)
        self.coarse_2_fine = fd.Function(self.scalar_space_fine)
        self.coarse_2_fine_original = fd.Function(self.scalar_space_fine)

        #       velocity field - the swirl: c
        self.c = fd.Function(self.vector_space)
        self.c_fine = fd.Function(self.vector_space_fine)
        self.cn = 0.5*(fd.dot(self.c, self.n) + abs(fd.dot(self.c, self.n)))
        self.cn_fine = 0.5*(fd.dot(self.c_fine, self.n_fine) + abs(fd.dot(self.c_fine, self.n_fine)))  # noqa

        # PDE problem RHS on coarse & fine mesh
        self.a = self.phi*self.du_trial*fd.dx(domain=self.mesh)
        self.a_fine = self.phi_fine*self.du_trial_fine*fd.dx(domain=self.mesh_fine)  # noqa

        # PDE problem LHS on coarse & fine mesh
        #       on coarse mesh
        self.L1 = self.dtc*(self.u*fd.div(self.phi*self.c)*fd.dx(domain=self.mesh)  # noqa
                - fd.conditional(fd.dot(self.c, self.n) < 0, self.phi*fd.dot(self.c, self.n)*self.u_in, 0.0)*fd.ds(domain=self.mesh)  # noqa
                - fd.conditional(fd.dot(self.c, self.n) > 0, self.phi*fd.dot(self.c, self.n)*self.u, 0.0)*fd.ds(domain=self.mesh)  # noqa
                - (self.phi('+') - self.phi('-'))*(self.cn('+')*self.u('+') - self.cn('-')*self.u('-'))*fd.dS(domain=self.mesh))  # noqa
        self.L2 = fd.replace(self.L1, {self.u: self.u1})
        self.L3 = fd.replace(self.L1, {self.u: self.u2})
        #       on fine mesh
        self.L1_fine = self.dtc*(self.u_fine*fd.div(self.phi_fine*self.c_fine)*fd.dx(domain=self.mesh_fine)  # noqa
                - fd.conditional(fd.dot(self.c_fine, self.n_fine) < 0, self.phi_fine*fd.dot(self.c_fine, self.n_fine)*self.u_in_fine, 0.0)*fd.ds(domain=self.mesh_fine)  # noqa
                - fd.conditional(fd.dot(self.c_fine, self.n_fine) > 0, self.phi_fine*fd.dot(self.c_fine, self.n_fine)*self.u_fine, 0.0)*fd.ds(domain=self.mesh_fine)  # noqa
                - (self.phi_fine('+') - self.phi_fine('-'))*(self.cn_fine('+')*self.u_fine('+') - self.cn_fine('-')*self.u_fine('-'))*fd.dS(domain=self.mesh_fine))  # noqa
        self.L2_fine = fd.replace(self.L1_fine, {self.u_fine: self.u1_fine})
        self.L3_fine = fd.replace(self.L1_fine, {self.u_fine: self.u2_fine})

        # vars for storing final solutions
        self.du = fd.Function(self.scalar_space)
        self.du_fine = fd.Function(self.scalar_space_fine)

        # PDE solver (one coarse & fine mesh) setup:
        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}  # noqa
        #       On coarse mesh
        self.prob1 = fd.LinearVariationalProblem(self.a, self.L1, self.du)
        self.solv1 = fd.LinearVariationalSolver(self.prob1, solver_parameters=params)  # noqa
        self.prob2 = fd.LinearVariationalProblem(self.a, self.L2, self.du)
        self.solv2 = fd.LinearVariationalSolver(self.prob2, solver_parameters=params)  # noqa
        self.prob3 = fd.LinearVariationalProblem(self.a, self.L3, self.du)
        self.solv3 = fd.LinearVariationalSolver(self.prob3, solver_parameters=params)  # noqa
        #       On fine mesh
        self.prob1_fine = fd.LinearVariationalProblem(self.a_fine, self.L1_fine, self.du_fine)  # noqa
        self.solv1_fine = fd.LinearVariationalSolver(self.prob1_fine, solver_parameters=params)  # noqa
        self.prob2_fine = fd.LinearVariationalProblem(self.a_fine, self.L2_fine, self.du_fine)  # noqa
        self.solv2_fine = fd.LinearVariationalSolver(self.prob2_fine, solver_parameters=params)  # noqa
        self.prob3_fine = fd.LinearVariationalProblem(self.a_fine, self.L3_fine, self.du_fine)  # noqa
        self.solv3_fine = fd.LinearVariationalSolver(self.prob3_fine, solver_parameters=params)  # noqa

    def solve_u(self, t):
        """
        Solve the PDE problem using RK (SSPRK) scheme on the coarse mesh
        store the solution field to a varaible: self.u_cur
        """
        c_exp = wm.get_c(self.x, self.y, t, alpha=self.alpha)
        c_temp = fd.Function(self.vector_space).interpolate(c_exp)
        self.c.project(c_temp)

        self.solv1.solve()
        self.u1.assign(self.u + self.du)

        self.solv2.solve()
        self.u2.assign(0.75*self.u + 0.25*(self.u1 + self.du))

        self.solv3.solve()
        self.u_cur.assign((1.0/3.0)*self.u + (2.0/3.0)*(self.u2 + self.du))

    def solve_u_fine(self, t):
        """
        Solve the PDE problem using RK (SSPRK) scheme on the fine mesh
        store the solution field to a varaible: self.u_cur_fine
        """
        c_exp = wm.get_c(self.x_fine, self.y_fine, t, alpha=self.alpha)
        c_temp = fd.Function(self.vector_space_fine).interpolate(c_exp)
        self.c_fine.project(c_temp)

        self.solv1_fine.solve()
        self.u1_fine.assign(self.u_fine + self.du_fine)

        self.solv2_fine.solve()
        self.u2_fine.assign(0.75*self.u_fine + 0.25*(self.u1_fine + self.du_fine))  # noqa

        self.solv3_fine.solve()
        self.u_cur_fine.assign((1.0/3.0)*self.u_fine + (2.0/3.0)*(self.u2_fine + self.du_fine))  # noqa

    def project_u_(self):
        self.u.project(self.u_fine_buffer)
        return

    def eval_problem(self):
        print("In eval problem")
        self.t = 0.0
        step = 0
        idx = 0
        for i in range(self.n_step):
            print(f"step: {step}, t: {self.t:.5f}")
            # error tracking lists init
            self.error_adapt_list = []
            self.error_og_list = []
            # data loading from raw file
            raw_data_path = self.dataset.file_names[idx]
            raw_data = np.load(raw_data_path, allow_pickle=True).item()
            data_t = raw_data.get('swirl_params')['t']
            y = raw_data.get('y')
            # solve PDE problem on fine mesh
            self.solve_u_fine(self.t)
            if (abs(self.t - data_t) < 1e-5):
                print(f"---- evaluating samples: step: {step}, t: {self.t:.5f}, data_t: {data_t:.5f}")  # noqa
                print("**", self.sigma, self.alpha, self.r_0, self.dt)
                print(data_t, self.t)
                # calculate solution on original mesh
                # self.mesh.coordinates.dat.data[:] = self.init_coord
                # self.project_u_()
                # self.solve_u(self.t)
                # function_space = fd.FunctionSpace(self.mesh, "CG", 1)
                # self.uh = fd.Function(function_space).project(self.u_cur)

                # calculate solution on adapted mesh
                self.adapt_coord = y
                # self.mesh.coordinates.dat.data[:] = self.adapt_coord
                # self.mesh_new.coordinates.dat.data[:] = self.adapt_coord
                # self.project_u_()
                # self.solve_u(self.t)
                # function_space_new = fd.FunctionSpace(self.mesh_new, "CG", 1)  # noqa
                # self.uh_new = fd.Function(function_space_new).project(self.u_cur)  # noqa

                # error measuring
                error_og, error_adapt = self.get_error()
                print(
                    f"error_og: {error_og}, \terror_adapt: {error_adapt}"
                )

                # plotting
                # plot = True
                # if plot is True:
                #     self.plot_res()
                #     plt.show()
                idx += 1

            # time stepping and prep for next solving iter
            self.t += self.dt
            step += 1
            self.u_fine.assign(self.u_cur_fine)
            self.u_fine_buffer.assign(self.u_cur_fine)
        return

    def get_error(self):
        # solve on fine mesh
        function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        self.solve_u_fine(self.t)
        u_fine = fd.Function(function_space_fine).project(self.u_cur_fine)  # noqa

        # solve on coarse mesh
        self.mesh.coordinates.dat.data[:] = self.init_coord
        self.project_u_()
        self.solve_u(self.t)
        u_og_2_fine = fd.project(self.u_cur, function_space_fine)

        # solve on coarse adapt mesh
        self.mesh.coordinates.dat.data[:] = self.adapt_coord
        self.project_u_()
        self.solve_u(self.t)
        u_adapt_2_fine = fd.project(self.u_cur, function_space_fine)

        # error calculation
        error_og = fd.errornorm(
            u_fine, u_og_2_fine, norm_type="L2"
        )
        error_adapt = fd.errornorm(
            u_fine,  u_adapt_2_fine, norm_type="L2"
        )

        # put mesh to init state
        self.mesh.coordinates.dat.data[:] = self.init_coord

        return error_og, error_adapt

    def plot_res(self):
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.set_title("Solution on fine mesh")
        fd.trisurf(self.u_cur_fine, axes=ax1)

        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        ax2.set_title("Solution on original mesh")
        fd.trisurf(self.uh, axes=ax2)

        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        ax3.set_title("Solution on adapt mesh")
        fd.trisurf(self.uh_new, axes=ax3)

        # ax4 = fig.add_subplot(2, 3, 4, projection="3d")
        # ax4.set_title("Hessian norm")
        # fd.trisurf(self.f_norm, axes=ax4)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title("Orignal mesh")
        fd.tripcolor(self.uh, axes=ax5, cmap='coolwarm')
        fd.triplot(self.mesh, axes=ax5)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title("adapted mesh")
        fd.tripcolor(self.uh_new, axes=ax6, cmap='coolwarm')
        fd.triplot(self.mesh_new, axes=ax6)

        return fig
