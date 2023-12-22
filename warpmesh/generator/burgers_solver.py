# Author: Chunyang Wang
# GitHub Username: chunyang-w

import firedrake as fd
import numpy as np                          # noqa
import matplotlib.pyplot as plt             # noqa
import movement as mv
import warpmesh as wm                       # noqa
import random                               # noqa

__all__ = ["BurgersSolver"]


class BurgersSolver():
    """
    Solves the Burgers equation
    Input:
    - mesh: The mesh on which to solve the equation.
    - dist_params: The parameters of the Gaussian distribution.

    """

    # def __init__(self, mesh, rand_generator, **kwargs):
    def __init__(self, mesh, **kwargs):
        """
        Initialise the solver.
        kwargs:
        - nu: The viscosity of the fluid.
        - dt: The time interval.
        """
        # Mesh
        self.mesh = mesh
        self.mesh_fine = fd.UnitSquareMesh(100, 100)

        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = self.mesh_fine.coordinates.vector().array().reshape(-1, 2) # noqa
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
        self.coarse_2_fine = fd.Function(self.P2_vec_fine)
        self.coarse_2_fine_original = fd.Function(self.P2_vec_fine)

        # simulation params
        self.nu = kwargs.pop("nu", 1e-3)
        self.gauss_list = kwargs.pop("gauss_list", None)
        self.dt = kwargs.get('dt', 1.0/30)
        self.sim_len = kwargs.get("T", 0.5)
        self.T = self.sim_len
        self.dtc = fd.Constant(self.dt)

        # # distribution params
        # self.u_init = rand_generator.get_u_exact(
        #     params={
        #         "x": self.x,
        #         "y": self.y,
        #     })
        # self.u_init_fine = rand_generator.get_u_exact(
        #     params={
        #         "x": self.x_fine,
        #         "y": self.y_fine,
        #     })
        # self.dist_params = rand_generator.get_dist_params()
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

        self.u_fine = fd.Function(self.P2_vec_fine)        # u_{0}
        self.u_fine_ = fd.Function(self.P2_vec_fine)           # u_{n+1}
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

        # solver params
        self.sp = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        # Solve for hessian
        self.normal = fd.FacetNormal(self.mesh)
        self.f_norm = fd.Function(self.P1)
        self.l2_projection = fd.Function(self.P1_ten)
        self.H, self.τ = fd.TrialFunction(
            self.P1_ten), fd.TestFunction(self.P1_ten)
        self.a = fd.inner(self.τ, self.H) * fd.dx(domain=self.mesh)

        self.L1 = -fd.inner(
            fd.div(self.τ), fd.grad(self.u[0])
        ) * fd.dx(domain=self.mesh)
        self.L1 += fd.dot(
            fd.grad(self.u[0]),
            fd.dot(self.τ, self.normal)
        ) * fd.ds(self.mesh)
        self.prob = fd.LinearVariationalProblem(
            self.a, self.L1, self.l2_projection)
        self.hessian_prob = fd.LinearVariationalSolver(
            self.prob, solver_parameters=self.sp
        )
        # Hessian norm - second dimension
        self.f_norm_2 = fd.Function(self.P1)  # comment:第二个分量的F范数啊
        self.l2_projection_2 = fd.Function(self.P1_ten)  # comment:第二个分量的F范数啊
        self.L2 = -fd.inner(
            fd.div(self.τ), fd.grad(self.u[1])
        ) * fd.dx(domain=self.mesh)
        self.L2 += fd.dot(
            fd.grad(self.u[1]),
            fd.dot(self.τ, self.normal)
        ) * fd.ds(self.mesh)
        self.prob2 = fd.LinearVariationalProblem(
            self.a, self.L2, self.l2_projection_2)
        self.hessian_prob2 = fd.LinearVariationalSolver(
            self.prob2, solver_parameters=self.sp)

    def monitor_function(self, mesh):
        if self.step == 1:
            self.u_.project(self.initial_velocity)
        else:
            self.u_.project(self.u_fine_buffer)
        fd.solve(self.F == 0, self.u)

        self.hessian_prob.solve()
        self.hessian_prob2.solve()

        self.f_norm.project(
            self.l2_projection[0, 0] ** 2 +
            self.l2_projection[0, 1] ** 2 +
            self.l2_projection[1, 0] ** 2 +
            self.l2_projection[1, 1] ** 2)

        self.f_norm_2.project(
            self.l2_projection_2[0, 0] ** 2 +
            self.l2_projection_2[0, 1] ** 2 +
            self.l2_projection_2[1, 0] ** 2 +
            self.l2_projection_2[1, 1] ** 2)

        # fig = plt.figure(figsize=(10, 5))
        # ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        # ax1.set_title('F_norm 1')
        # fd.trisurf(self.f_norm, axes=ax1)
        # ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        # ax2.set_title('F_norm 2')
        # fd.trisurf(self.f_norm_2, axes=ax2)
        # plt.show()

        max_1 = self.f_norm.vector().max()
        max_2 = self.f_norm_2.vector().max()
        if max_1 >= max_2:
            self.f_norm /= max_1
            self.f_norm_2 /= max_1
            self.f_norm.assign(self.f_norm + self.f_norm_2)
        else:
            self.f_norm /= max_2
            self.f_norm_2 /= max_2
            self.f_norm.assign(self.f_norm + self.f_norm_2)
        self.f_norm /= self.f_norm.vector().max()
        monitor = self.f_norm
        return 1 + (5 * monitor)

    def solve_problem(self):
        """
        Solves the Burgers equation.
        """
        t = 0.0
        self.step = 0
        # output_freq = 5
        # while t < self.T - 0.5*self.dt:
        for i in range(10):
            self.step += 1
            print("step: {}, t: {}".format(self.step, t))
            fd.solve(self.F_fine == 0, self.u_fine)
            fd.solve(self.F == 0, self.u)
            self.u_fine_.assign(self.u_fine)
            adapter = mv.MongeAmpereMover(
                self.mesh,
                monitor_function=self.monitor_function,
                rtol=1e-3,
            )
            adapter.move()
            self.u_fine_buffer.project(self.u)
            self.coarse_2_fine.assign(self.u_fine_buffer)

            self.u_.assign(self.u_og)
            fd.solve(self.F == 0, self.u)
            self.u_og.assign(self.u)
            self.coarse_2_fine_original.project(self.u_og)

            # fig = plt.figure(figsize=(8, 8))
            # ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            # ax1.set_title('f_norm')
            # fd.trisurf(self.f_norm, axes=ax1)
            # ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            # ax2.set_title('solution')
            # fd.trisurf(self.u, axes=ax2)
            # ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            # ax3.set_title('Hessian')
            # fd.trisurf(self.f_norm, axes=ax3)
            # ax4 = fig.add_subplot(2, 2, 4)
            # ax4.set_title('Mesh')
            # ax4.set_aspect('equal')
            # fd.tripcolor(self.u, axes=ax4, cmap='coolwarm')
            # # fd.tricontour(self.u, axes=ax4, cmap='coolwarm')
            # fd.triplot(self.mesh, axes=ax4)

            # plt.show()

            self.mesh.coordinates.dat.data[:] = self.init_coord

            t += self.dt
        plt.show()


def get_sample_param_of_nu_generalization_by_idx_train(idx_in):
    gauss_list_ = []
    if idx_in == 1:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 2:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 3:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 4:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 5:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 6:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 7:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 8:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 9:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    return gauss_list_, nu_


if __name__ == "__main__":
    # parameters for anisotropic data - distribution height scaler
    z_min = 0
    z_max = 1

    # parameters for isotropic data
    w_min = 0.05
    w_max = 0.2

    c_min = 0.2
    c_max = 0.8

    gaussian_list, nu = get_sample_param_of_nu_generalization_by_idx_train(1)

    mesh = fd.UnitSquareMesh(32, 32)

    solver = BurgersSolver(mesh, gauss_list=gaussian_list, nu=nu)
    solver.solve_problem()
