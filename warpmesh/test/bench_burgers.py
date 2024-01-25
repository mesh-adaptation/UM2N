# Author: Chunyang Wang
# GitHub Username: chunyang-w

import firedrake as fd
import numpy as np                          # noqa
import matplotlib.pyplot as plt             # noqa
import os                                   # noqa
import random                               # noqa
import time                                 # noqa
import warpmesh as wm                       # noqa


class BurgersEvaluator():
    """
    Solves the Burgers equation
    Input:
    - mesh: The mesh on which to solve the equation.
    - dist_params: The parameters of the Gaussian distribution.

    """
    def __init__(self, mesh, mesh_fine, mesh_new,
                 dataset, model, eval_dir, ds_root, idx,
                 **kwargs):  # noqa
        """
        Initialise the solver.
        kwargs:
        - nu: The viscosity of the fluid.
        - dt: The time interval.
        """
        # Mesh
        self.mesh = mesh
        self.mesh_fine = mesh_fine
        self.mesh_new = mesh_new
        # evaluation vars
        self.dataset = dataset              # dataset containing all data
        self.model = model                  # the NN model
        self.eval_dir = eval_dir            # evaluation root dir
        self.ds_root = ds_root
        self.log_path = os.path.join(eval_dir, "log")
        self.plot_path = os.path.join(eval_dir, "plot")
        self.plot_more_path = os.path.join(eval_dir, "plot_more")
        self.idx = idx
        # coordinates
        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = self.mesh_fine.coordinates.vector().array().reshape(-1, 2) # noqa
        self.best_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.adapt_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        self.error_adapt_list = []
        self.error_og_list = []
        self.best_error_iter = 0

        # X and Y coordinates
        self.x, self.y = fd.SpatialCoordinate(mesh)
        self.x_fine, self.y_fine = fd.SpatialCoordinate(self.mesh_fine)
        # Function spaces
        self.P1 = fd.FunctionSpace(mesh, "CG", 1)
        self.P2 = fd.FunctionSpace(mesh, "CG", 2)
        self.P1_vec = fd.VectorFunctionSpace(mesh, "CG", 1)
        self.P2_vec = fd.VectorFunctionSpace(mesh, "CG", 2)
        self.P1_ten = fd.TensorFunctionSpace(mesh, "CG", 1)
        self.P2_ten = fd.TensorFunctionSpace(mesh, "CG", 2)

        self.P1_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        self.P2_vec_fine = fd.VectorFunctionSpace(self.mesh_fine, "CG", 2)
        self.phi_p2_vec_fine = fd.TestFunction(self.P2_vec_fine)

        # Test functions
        self.phi = fd.TestFunction(self.P1)
        self.phi_p2_vec = fd.TestFunction(self.P2_vec)

        self.trial_fine = fd.TrialFunction(self.P1_fine)
        self.phi_fine = fd.TestFunction(self.P1_fine)

        # buffer
        self.u_fine_buffer = fd.Function(self.P2_vec_fine)
        self.coarse_adapt = fd.Function(self.P1_vec)
        self.coarse_2_fine = fd.Function(self.P2_vec_fine)
        self.coarse_2_fine_original = fd.Function(self.P2_vec_fine)

        # simulation params
        self.nu = kwargs.pop("nu", 1e-3)
        self.gauss_list = kwargs.pop("gauss_list", None)
        self.dt = kwargs.get('dt', 1.0/30)
        self.sim_len = kwargs.get("T", 2.0)
        self.T = self.sim_len
        self.dtc = fd.Constant(self.dt)

        self.u_init = 0
        self.u_init_fine = 0
        num_of_gauss = len(self.gauss_list)
        for counter in range(num_of_gauss):
            c_x, c_y, w = self.gauss_list[counter]["cx"], self.gauss_list[counter]["cy"], self.gauss_list[counter]["w"]  # noqa
            self.u_init += fd.exp(-((self.x - c_x) ** 2 + (self.y - c_y) ** 2) / w)  # noqa
            self.u_init_fine += fd.exp(-((self.x_fine - c_x) ** 2 + (self.y_fine - c_y) ** 2) / w)  # noqa

        # solution vars
        self.u_og = fd.Function(self.P2_vec)        # u_{0}
        self.u = fd.Function(self.P2_vec)           # u_{n+1}
        self.u_ = fd.Function(self.P2_vec)          # u_{n}
        self.F = (
            fd.inner(
                (self.u - self.u_) / self.dtc, self.phi_p2_vec
                ) +
            fd.inner(
                fd.dot(self.u, fd.nabla_grad(self.u)), self.phi_p2_vec
                ) +
            self.nu * fd.inner(
                fd.grad(self.u), fd.grad(self.phi_p2_vec)
                )
            ) * fd.dx(domain=self.mesh)

        self.u_fine = fd.Function(self.P2_vec_fine)             # u_{0}
        self.u_fine_ = fd.Function(self.P2_vec_fine)            # u_{n+1}
        self.F_fine = (
            fd.inner(
                (self.u_fine - self.u_fine_) / self.dtc, self.phi_p2_vec_fine
                ) +
            fd.inner(
                fd.dot(self.u_fine, fd.nabla_grad(self.u_fine)), self.phi_p2_vec_fine  # noqa
                ) +
            self.nu * fd.inner(
                fd.grad(self.u_fine), fd.grad(self.phi_p2_vec_fine)
                )
            ) * fd.dx(domain=self.mesh_fine)

        # initial vals
        self.initial_velocity = fd.as_vector([self.u_init, 0])
        self.initial_velocity_fine = fd.as_vector([self.u_init_fine, 0])

        self.u.project(self.initial_velocity)
        self.u_.assign(self.u)
        self.u_og.assign(self.u)

        ic_fine = fd.project(self.initial_velocity_fine, self.P2_vec_fine)
        self.u_fine.assign(ic_fine)
        self.u_fine_.assign(ic_fine)
        self.u_fine_buffer.assign(ic_fine)

        # solver params
        self.sp = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    def project_u_(self):
        self.u_.project(self.u_fine_buffer)

    def eval_problem(self):
        """
        Solves the Burgers equation.
        """
        idx_start = 60 * (self.idx - 1)
        i = 0
        t = 0.0
        self.step = 0
        self.best_error_iter = 0
        while t < self.T - 0.5*self.dt:
            # get sample for item
            cur_step = idx_start + i
            sample = self.dataset[cur_step]
            print("cur_step: ", cur_step)
            self.error_adapt_list = []
            self.error_og_list = []
            mesh_new = self.mesh_new

            print("step: {}, t: {}".format(self.step, t))
            # solve on fine mesh
            fd.solve(self.F_fine == 0, self.u_fine)

            self.adapt_coord = sample.y
            mesh_new.coordinates.dat.data[:] = self.adapt_coord

            # calculate solution on original mesh
            self.mesh.coordinates.dat.data[:] = self.init_coord
            self.project_u_()
            fd.solve(self.F == 0, self.u)
            function_space = fd.FunctionSpace(self.mesh, "CG", 1)
            uh_0 = fd.Function(function_space)
            uh_0.project(self.u[0])

            # calculate solution on adapted mesh
            self.mesh.coordinates.dat.data[:] = self.adapt_coord
            self.project_u_()
            fd.solve(self.F == 0, self.u)
            function_space_new = fd.FunctionSpace(mesh_new, "CG", 1)
            function_space_vec_new = fd.VectorFunctionSpace(
                mesh_new, "CG", 1)
            uh_new = fd.Function(function_space_vec_new)
            uh_new.project(self.u)
            uh_new_0 = fd.Function(function_space_new)
            uh_new_0.project(uh_new[0])

            error_og, error_adapt = self.get_error()
            print(
                "error_og: {}, error_adapt: {}".format(error_og, error_adapt))

            # put coords back to original position (for u sampling)
            self.mesh.coordinates.dat.data[:] = self.init_coord

            # step forward in time
            self.u_fine_.assign(self.u_fine)
            # self.u_fine_buffer.project(self.u)
            self.u_fine_buffer.assign(self.u_fine)
            fd.triplot(self.u_fine)
            plt.show()
            # self.u_.assign(self.u)
            t += self.dt
            self.step += 1
            i += 1
        return

    def get_error(self):
        # print("get_error: u_ sum is: ", np.sum(self.u_.dat.data[:]))
        function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        # solve on fine mesh
        fd.solve(self.F_fine == 0, self.u_fine)
        u_fine_0 = fd.Function(function_space_fine)
        u_f = u_fine_0.project(self.u_fine[0])
        # print('u_f sum: ', np.sum(u_f.dat.data[:]))

        # solve on coarse mesh
        self.mesh.coordinates.dat.data[:] = self.init_coord
        function_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.project_u_()
        # print("og u_ sum: ", np.sum(self.u_.dat.data[:]))
        fd.solve(self.F == 0, self.u)
        u_0_fine = fd.Function(function_space_fine)
        u_0_coarse = fd.Function(function_space)
        u_0_coarse.project(self.u[0])
        u_0_fine.project(u_0_coarse)
        # print('u_0_fine sum 1: ', np.sum(u_0_fine.dat.data[:]))
        error_og = fd.errornorm(
            u_0_fine, u_f, norm_type="L2")

        # solve on coarse adapt mesh
        self.mesh.coordinates.dat.data[:] = self.adapt_coord
        function_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.project_u_()
        # print("adapt u_ sum: ", np.sum(self.u_.dat.data[:]))
        fd.solve(self.F == 0, self.u)
        u_adapt_fine_0 = fd.Function(function_space_fine)
        u_adapt_coarse_0 = fd.Function(function_space)
        u_adapt_coarse_0.project(self.u[0])
        u_adapt_fine_0.project(u_adapt_coarse_0)
        # print('u sum 2: ', np.sum(u_adapt_fine_0.dat.data[:]))
        error_adapt = fd.errornorm(
            u_adapt_fine_0, u_f, norm_type="L2")

        self.mesh.coordinates.dat.data[:] = self.init_coord

        return error_og, error_adapt

    def make_log_dir(self):
        wm.mkdir_if_not_exist(self.log_path)

    def make_plot_dir(self):
        wm.mkdir_if_not_exist(self.plot_path)

    def make_plot_more_dir(self):
        wm.mkdir_if_not_exist(self.plot_more_path)