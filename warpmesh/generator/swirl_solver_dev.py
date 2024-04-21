# Author: Chunyang Wang, Mingrui Zhang
# GitHub Username: chunyang-w
# Description: Solve advection swirl problem

import time  # noqa

import firedrake as fd  # noqa

import movement as mv  # noqa
import warpmesh as wm  # noqa
import numpy as np

import matplotlib.pyplot as plt  # noqa

from tqdm import tqdm  # noqa


def get_c(x, y, t, threshold=0.5, alpha=1.5):
    """
    Compute the velocity field which transports the
    solution field u.
    Args:
        alpha (float): coefficient for velocity magnitude.

    Return:
        velocity (ufl.tensors): expression of the swirl velocity field
    """
    a = 1 if t < threshold else -1
    v_x = alpha * a * fd.sin(fd.pi * x) ** 2 * fd.sin(2 * fd.pi * y)
    v_y = -1 * alpha * a * fd.sin(fd.pi * y) ** 2 * fd.sin(2 * fd.pi * x)
    velocity = fd.as_vector((v_x, v_y))
    return velocity


def get_u_0(x, y, r_0=0.2, x_0=0.5, y_0=0.75, sigma=(0.05 / 3)):
    """
    Compute the initial trace value.

    Return:
        u_0 (ufl.tensors): expression of u_0
    """
    u = fd.exp(
        (-1 / (2 * sigma**2)) * (fd.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) - r_0) ** 2
    )
    return u


class SwirlSolver:
    """
    Solver for advection swirl problem:
        1. Solver implementation for the swirl problem
        2. Mesh mover for the swirl problem
        3. Error & Time evaluation
    """

    def __init__(self, mesh, mesh_fine, mesh_new, **kwargs):
        """
        Init the problem:
            1. define problem on fine mesh and coarse mesh
            2. init function space on fine & coarse mesh
            3. define hessian solver on coarse mesh
        """
        self.mesh = mesh  # coarse mesh
        self.mesh_fine = mesh_fine  # fine mesh
        self.mesh_new = mesh_new  # adapted mesh
        self.save_interval = kwargs.pop("save_interval", 5)
        self.n_monitor_smooth = kwargs.pop("n_monitor_smooth", 10)
        # Init coords setup
        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = (
            self.mesh_fine.coordinates.vector().array().reshape(-1, 2)
        )  # noqa
        self.best_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.adapt_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        self.adapt_coord_prev = (
            self.mesh.coordinates.vector().array().reshape(-1, 2)
        )  # noqa
        # error measuring vars
        self.error_adapt_list = []
        self.error_og_list = []
        self.best_error_iter = 0

        # X and Y coordinates
        self.x, self.y = fd.SpatialCoordinate(mesh)
        self.x_fine, self.y_fine = fd.SpatialCoordinate(self.mesh_fine)

        # function space on coarse mesh
        # self.scalar_space = fd.FunctionSpace(self.mesh, "CG", 1)
        # self.vector_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        # self.tensor_space = fd.TensorFunctionSpace(self.mesh, "CG", 1)
        # # function space on fine mesh
        # self.scalar_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        # self.vector_space_fine = fd.VectorFunctionSpace(self.mesh_fine, "CG", 1)  # noqa

        self.scalar_space = fd.FunctionSpace(self.mesh, "DG", 1)
        self.vector_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        self.tensor_space = fd.TensorFunctionSpace(self.mesh, "DG", 1)
        # function space on fine mesh
        self.scalar_space_fine = fd.FunctionSpace(self.mesh_fine, "DG", 1)
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
        self.n_step = kwargs.pop("n_step", 1000)
        self.threshold = (
            self.T / 2
        )  # Time point the swirl direction get reverted  # noqa
        self.dt = self.T / self.n_step
        self.dtc = fd.Constant(self.dt)
        # initial condition params
        self.sigma = kwargs.pop("sigma", (0.05 / 6))
        self.alpha = kwargs.pop("alpha", 1.5)
        self.r_0 = kwargs.pop("r_0", 0.2)
        self.x_0 = kwargs.pop("x_0", 0.25)
        self.y_0 = kwargs.pop("y_0", 0.25)

        # initital condition of u on coarse / fine mesh
        u_init_exp = get_u_0(
            self.x, self.y, self.r_0, self.x_0, self.y_0, self.sigma
        )  # noqa
        u_init_exp_fine = get_u_0(
            self.x_fine, self.y_fine, self.r_0, self.x_0, self.y_0, self.sigma
        )  # noqa
        self.u_init = fd.Function(self.scalar_space).interpolate(u_init_exp)
        self.u_init_fine = fd.Function(self.scalar_space_fine).interpolate(
            u_init_exp_fine
        )  # noqa
        # PDE vars on coarse & fine mesh
        #       solution field u
        self.u = fd.Function(self.scalar_space).assign(self.u_init)
        self.u1 = fd.Function(self.scalar_space)
        self.u2 = fd.Function(self.scalar_space)
        self.u_fine = fd.Function(self.scalar_space_fine).assign(
            self.u_init_fine
        )  # noqa
        self.u1_fine = fd.Function(self.scalar_space_fine)
        self.u2_fine = fd.Function(self.scalar_space_fine)
        self.u_in = fd.Constant(0.0)
        self.u_in_fine = fd.Constant(0.0)
        #       temp vars for saving u on coarse & fine mesh
        self.u_cur = fd.Function(
            self.scalar_space
        )  # solution from current time step  # noqa
        self.u_cur_fine = fd.Function(self.scalar_space_fine)
        self.u_hess = fd.Function(
            self.scalar_space
        )  # buffer for hessian solver usage  # noqa
        #       buffers
        self.u_fine_buffer = fd.Function(self.scalar_space_fine).assign(
            self.u_init_fine
        )  # noqa
        self.coarse_adapt = fd.Function(self.scalar_space)
        self.coarse_2_fine = fd.Function(self.scalar_space_fine)
        self.coarse_2_fine_original = fd.Function(self.scalar_space_fine)

        #       velocity field - the swirl: c
        self.c = fd.Function(self.vector_space)
        self.c_fine = fd.Function(self.vector_space_fine)
        self.cn = 0.5 * (fd.dot(self.c, self.n) + abs(fd.dot(self.c, self.n)))
        self.cn_fine = 0.5 * (
            fd.dot(self.c_fine, self.n_fine) + abs(fd.dot(self.c_fine, self.n_fine))
        )  # noqa

        # PDE problem RHS on coarse & fine mesh
        self.a = self.phi * self.du_trial * fd.dx(domain=self.mesh)
        self.a_fine = (
            self.phi_fine * self.du_trial_fine * fd.dx(domain=self.mesh_fine)
        )  # noqa

        # PDE problem LHS on coarse & fine mesh
        #       on coarse mesh
        self.L1 = self.dtc * (
            self.u * fd.div(self.phi * self.c) * fd.dx(domain=self.mesh)  # noqa
            - fd.conditional(
                fd.dot(self.c, self.n) < 0,
                self.phi * fd.dot(self.c, self.n) * self.u_in,
                0.0,
            )
            * fd.ds(domain=self.mesh)  # noqa
            - fd.conditional(
                fd.dot(self.c, self.n) > 0,
                self.phi * fd.dot(self.c, self.n) * self.u,
                0.0,
            )
            * fd.ds(domain=self.mesh)  # noqa
            - (self.phi("+") - self.phi("-"))
            * (self.cn("+") * self.u("+") - self.cn("-") * self.u("-"))
            * fd.dS(domain=self.mesh)
        )  # noqa
        self.L2 = fd.replace(self.L1, {self.u: self.u1})
        self.L3 = fd.replace(self.L1, {self.u: self.u2})
        #       on fine mesh
        self.L1_fine = self.dtc * (
            self.u_fine
            * fd.div(self.phi_fine * self.c_fine)
            * fd.dx(domain=self.mesh_fine)  # noqa
            - fd.conditional(
                fd.dot(self.c_fine, self.n_fine) < 0,
                self.phi_fine * fd.dot(self.c_fine, self.n_fine) * self.u_in_fine,
                0.0,
            )
            * fd.ds(domain=self.mesh_fine)  # noqa
            - fd.conditional(
                fd.dot(self.c_fine, self.n_fine) > 0,
                self.phi_fine * fd.dot(self.c_fine, self.n_fine) * self.u_fine,
                0.0,
            )
            * fd.ds(domain=self.mesh_fine)  # noqa
            - (self.phi_fine("+") - self.phi_fine("-"))
            * (
                self.cn_fine("+") * self.u_fine("+")
                - self.cn_fine("-") * self.u_fine("-")
            )
            * fd.dS(domain=self.mesh_fine)
        )  # noqa
        self.L2_fine = fd.replace(self.L1_fine, {self.u_fine: self.u1_fine})
        self.L3_fine = fd.replace(self.L1_fine, {self.u_fine: self.u2_fine})

        # vars for storing final solutions
        self.du = fd.Function(self.scalar_space)
        self.du_fine = fd.Function(self.scalar_space_fine)

        # PDE solver (one coarse & fine mesh) setup:
        params = {
            "ksp_type": "preonly",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }  # noqa
        #       On coarse mesh
        self.prob1 = fd.LinearVariationalProblem(self.a, self.L1, self.du)
        self.solv1 = fd.LinearVariationalSolver(
            self.prob1, solver_parameters=params
        )  # noqa
        self.prob2 = fd.LinearVariationalProblem(self.a, self.L2, self.du)
        self.solv2 = fd.LinearVariationalSolver(
            self.prob2, solver_parameters=params
        )  # noqa
        self.prob3 = fd.LinearVariationalProblem(self.a, self.L3, self.du)
        self.solv3 = fd.LinearVariationalSolver(
            self.prob3, solver_parameters=params
        )  # noqa
        #       On fine mesh
        self.prob1_fine = fd.LinearVariationalProblem(
            self.a_fine, self.L1_fine, self.du_fine
        )  # noqa
        self.solv1_fine = fd.LinearVariationalSolver(
            self.prob1_fine, solver_parameters=params
        )  # noqa
        self.prob2_fine = fd.LinearVariationalProblem(
            self.a_fine, self.L2_fine, self.du_fine
        )  # noqa
        self.solv2_fine = fd.LinearVariationalSolver(
            self.prob2_fine, solver_parameters=params
        )  # noqa
        self.prob3_fine = fd.LinearVariationalProblem(
            self.a_fine, self.L3_fine, self.du_fine
        )  # noqa
        self.solv3_fine = fd.LinearVariationalSolver(
            self.prob3_fine, solver_parameters=params
        )  # noqa

        # Monitor function variables
        self.grad_norm = fd.Function(self.scalar_space)
        self.monitor_values = fd.Function(self.scalar_space)
        # Hessian solver
        hess_param = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }  # noqa
        self.normal = fd.FacetNormal(self.mesh)
        self.f_norm = fd.Function(self.scalar_space)
        self.l2_projection = fd.Function(self.tensor_space)
        self.H, self.τ = fd.TrialFunction(self.tensor_space), fd.TestFunction(
            self.tensor_space
        )
        #     LHS & RHS
        self.a_hess = fd.inner(self.τ, self.H) * fd.dx(domain=self.mesh)
        self.L1_hess = -fd.inner(fd.div(self.τ), fd.grad(self.u_hess)) * fd.dx(
            domain=self.mesh
        )
        self.L1_hess += fd.dot(
            fd.grad(self.u_hess), fd.dot(self.τ, self.normal)
        ) * fd.ds(self.mesh)
        self.prob_hess = fd.LinearVariationalProblem(
            self.a_hess, self.L1_hess, self.l2_projection
        )
        self.hessian_prob = fd.LinearVariationalSolver(
            self.prob_hess, solver_parameters=hess_param
        )

    def solve_u(self, t):
        """
        Solve the PDE problem using RK (SSPRK) scheme on the coarse mesh
        store the solution field to a varaible: self.u_cur
        """
        c_exp = get_c(self.x, self.y, t, alpha=self.alpha)
        # c_temp = fd.Function(self.vector_space).interpolate(c_exp)
        # self.c.project(c_temp)
        self.c.interpolate(c_exp)

        self.solv1.solve()
        self.u1.assign(self.u + self.du)

        self.solv2.solve()
        self.u2.assign(0.75 * self.u + 0.25 * (self.u1 + self.du))

        self.solv3.solve()
        self.u_cur.assign((1.0 / 3.0) * self.u + (2.0 / 3.0) * (self.u2 + self.du))

    def solve_u_fine(self, t):
        """
        Solve the PDE problem using RK (SSPRK) scheme on the fine mesh
        store the solution field to a varaible: self.u_cur_fine
        """
        c_exp = get_c(self.x_fine, self.y_fine, t, alpha=self.alpha)
        # c_temp = fd.Function(self.vector_space_fine).interpolate(c_exp)
        # self.c_fine.project(c_temp)
        self.c_fine.interpolate(c_exp)

        self.solv1_fine.solve()
        self.u1_fine.assign(self.u_fine + self.du_fine)

        self.solv2_fine.solve()
        self.u2_fine.assign(
            0.75 * self.u_fine + 0.25 * (self.u1_fine + self.du_fine)
        )  # noqa

        self.solv3_fine.solve()
        self.u_cur_fine.assign(
            (1.0 / 3.0) * self.u_fine + (2.0 / 3.0) * (self.u2_fine + self.du_fine)
        )  # noqa

    def project_u_(self):
        self.u.project(self.u_fine_buffer)
        return

    def monitor_function_pure_hessian(self, mesh, beta=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa

        self.monitor_values.project(1 + beta * self.f_norm)
        return self.monitor_values

    def monitor_function_smoothed_hessian(self, mesh, beta=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        self.f_norm.dat.data[:] = 1 / (1 + np.exp(-self.f_norm.dat.data[:])) - 0.5
        self.f_norm /= self.f_norm.vector().max()

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa

        self.monitor_values.project(1 + beta * self.f_norm)
        return self.monitor_values

    def monitor_function_grad(self, mesh, alpha=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa

        self.monitor_values.project(1 + alpha * self.grad_norm)
        return self.monitor_values

    def monitor_function_smoothed_grad(self, mesh, alpha=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()
        self.grad_norm.dat.data[:] = 1 / (1 + np.exp(-self.grad_norm.dat.data[:])) - 0.5
        self.grad_norm /= self.grad_norm.vector().max()

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa

        self.monitor_values.project(1 + alpha * self.grad_norm)
        return self.monitor_values

    def monitor_function(self, mesh, alpha=10, beta=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        self.f_norm.dat.data[:] = 1 / (1 + np.exp(-self.f_norm.dat.data[:])) - 0.5
        self.f_norm /= self.f_norm.vector().max()

        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()
        self.grad_norm.dat.data[:] = 1 / (1 + np.exp(-self.grad_norm.dat.data[:])) - 0.5
        self.grad_norm /= self.grad_norm.vector().max()

        # Interpolate on P0 space and then project back to P1 to induce numerical diffusion
        # p0_space = fd.FunctionSpace(self.mesh, "CG", 0)
        # self.f_norm = fd.interpolate(self.f_norm, p0_space).project(self.f_norm)
        # self.grad_norm = fd.interpolate(self.grad_norm, p0_space).project(self.grad_norm)

        # Choose the max values between grad norm and hessian norm according to
        # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
        # self.monitor_values.dat.data[:] = np.maximum(
        #     beta * self.f_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
        # )

        self.monitor_values.dat.data[:] = (
            beta * self.f_norm.dat.data[:] + alpha * self.grad_norm.dat.data[:]
        ) / 2

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa

        self.monitor_values.project(1 + self.monitor_values)

        return self.monitor_values

    def monitor_function_for_merge(self, mesh, alpha=10, beta=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()

        # Choose the max values between grad norm and hessian norm according to
        # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
        self.monitor_values.dat.data[:] = np.maximum(
            beta * self.f_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
        )

        # #################

        # V = fd.FunctionSpace(mesh, "CG", 1)
        # u = fd.TrialFunction(V)
        # v = fd.TestFunction(V)
        # function_space = V
        # # Discretised Eq Definition Start
        # f = self.monitor_values
        # N = 40  # As suggested in eq 23
        # dx = 1 / 35
        # K = N * dx**2 / 4
        # RHS = f * v * fd.dx(domain=mesh)
        # LHS = (K * fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        # bc = fd.DirichletBC(function_space, f, "on_boundary")

        # monitor_smoothed = fd.Function(function_space)
        # fd.solve(
        #     LHS == RHS,
        #     monitor_smoothed,
        #     solver_parameters={"ksp_type": "cg", "pc_type": "none"},
        #     bcs=bc,
        # )

        # #################

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        self.monitor_values.project(1 + self.monitor_values)

        return self.monitor_values

    def monitor_function(self, mesh, alpha=10, beta=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()

        # Choose the max values between grad norm and hessian norm according to
        # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
        self.monitor_values.dat.data[:] = np.maximum(
            beta * self.f_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
        )

        # #################

        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        function_space = V
        # Discretised Eq Definition Start
        f = self.monitor_values
        dx = mesh.cell_sizes.dat.data[:].mean()
        N = self.n_monitor_smooth
        K = N * dx**2 / 4
        RHS = f * v * fd.dx(domain=mesh)
        LHS = (K * fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        bc = fd.DirichletBC(function_space, f, "on_boundary")

        monitor_smoothed = fd.Function(function_space)
        fd.solve(
            LHS == RHS,
            monitor_smoothed,
            solver_parameters={"ksp_type": "cg", "pc_type": "none"},
            bcs=bc,
        )

        # #################

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        self.monitor_values.project(1 + monitor_smoothed)

        return self.monitor_values

    def monitor_function_on_coarse_mesh(self, mesh, alpha=10, beta=5):
        self.project_u_()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.hessian_prob.solve()
        self.f_norm.project(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()

        monitor_values = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        # Choose the max values between grad norm and hessian norm according to
        # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
        monitor_values.dat.data[:] = np.maximum(
            beta * self.f_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
        )

        # #################

        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        function_space = V
        # Discretised Eq Definition Start
        f = monitor_values
        dx = mesh.cell_sizes.dat.data[:].mean()
        N = self.n_monitor_smooth
        K = N * dx**2 / 4
        RHS = f * v * fd.dx(domain=mesh)
        LHS = (K * fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        bc = fd.DirichletBC(function_space, f, "on_boundary")

        monitor_smoothed = fd.Function(function_space)
        fd.solve(
            LHS == RHS,
            monitor_smoothed,
            solver_parameters={"ksp_type": "cg", "pc_type": "none"},
            bcs=bc,
        )

        # #################
        monitor_values.project(1 + monitor_smoothed)

        return monitor_values

    def solve_problem(self, callback=None, fail_callback=None):
        print("In solve problem")
        self.t = 0.0
        step = 0
        for i in range(self.n_step):
            print(f"step: {step}, t: {self.t:.5f}")
            # error tracking lists init
            self.error_adapt_list = []
            self.error_og_list = []
            # solve PDE problem on fine mesh
            self.solve_u_fine(self.t)
            # if ((step + 1) % self.save_interval == 0) or (step == 0):
            #     print(f"---- getting samples: step: {step}, t: {self.t:.5f}")
            #     try:

            #         monitor_values = self.monitor_function_on_coarse_mesh(self.mesh)
            #         hessian_norm = self.f_norm
            #         grad_u_norm = self.grad_norm

            #         # # TODO: Starting the mesh movement from last adapted mesh (This is not working currently)
            #         # self.mesh.coordinates.dat.data[:] = self.adapt_coord_prev
            #         # mesh movement - calculate the adapted coords
            #         start = time.perf_counter()
            #         adapter = mv.MongeAmpereMover(
            #             self.mesh, monitor_function=self.monitor_function, rtol=1e-3, maxiter=100
            #         )
            #         adapter.move()
            #         end = time.perf_counter()
            #         dur_ms = (end - start) * 1e3
            #         self.mesh_new.coordinates.dat.data[:] = self.adapt_coord
            #         # self.adapt_coord_prev = self.mesh_new.coordinates.dat.data[:]

            #         # calculate solution on original mesh
            #         self.mesh.coordinates.dat.data[:] = self.init_coord
            #         self.project_u_()
            #         self.solve_u(self.t)
            #         function_space = fd.FunctionSpace(self.mesh, "CG", 1)
            #         self.uh = fd.Function(function_space).project(self.u_cur)

            #         # calculate solution on adapted mesh
            #         self.mesh.coordinates.dat.data[:] = self.adapt_coord
            #         self.project_u_()
            #         self.solve_u(self.t)
            #         function_space_new = fd.FunctionSpace(
            #             self.mesh_new, "CG", 1
            #         )  # noqa
            #         self.uh_new = fd.Function(function_space_new).project(
            #             self.u_cur
            #         )  # noqa

            #         # error measuring
            #         error_og, error_adapt = self.get_error()
            #         print(f"error_og: {error_og}, \terror_adapt: {error_adapt}")

            #         # put coords back to init state and sampling for datasets
            #         self.mesh.coordinates.dat.data[:] = self.init_coord

            #         # plotting
            #         plot = False
            #         if plot is True:
            #             self.plot_res()
            #             plt.show()

            #         # retrive info from original mesh and save data
            #         function_space = fd.FunctionSpace(self.mesh, "CG", 1)
            #         function_space_fine = fd.FunctionSpace(
            #             self.mesh_fine, "CG", 1
            #         )  # noqa
            #         uh_fine = fd.Function(function_space_fine)
            #         uh_fine.project(self.u_cur_fine)

            #         func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
            #         uh_grad = fd.interpolate(fd.grad(self.uh), func_vec_space)
            #         # hessian_norm = self.f_norm
            #         # monitor_values = adapter.monitor
            #         hessian = self.l2_projection
            #         phi = adapter.phi
            #         phi_grad = adapter.grad_phi
            #         sigma = adapter.sigma
            #         I = fd.Identity(2)  # noqa
            #         jacobian = I + sigma
            #         jacobian_det = fd.Function(function_space, name="jacobian_det")
            #         jacobian_det.project(
            #             jacobian[0, 0] * jacobian[1, 1]
            #             - jacobian[0, 1] * jacobian[1, 0]
            #         )
            #         self.jacob_det = fd.project(
            #             jacobian_det, fd.FunctionSpace(self.mesh, "CG", 1)
            #         )
            #         self.jacob = fd.project(
            #             jacobian, fd.TensorFunctionSpace(self.mesh, "CG", 1)
            #         )
            #         callback(
            #             uh=self.uh,
            #             uh_grad=uh_grad,
            #             grad_u_norm=grad_u_norm,
            #             hessian_norm=hessian_norm,
            #             monitor_values=monitor_values,
            #             hessian=hessian,
            #             phi=phi,
            #             grad_phi=phi_grad,
            #             jacobian=self.jacob,
            #             jacobian_det=self.jacob_det,
            #             mesh_new=self.mesh_new,
            #             mesh_og=self.mesh,
            #             uh_new=self.uh_new,
            #             uh_fine=uh_fine,
            #             function_space=function_space,
            #             function_space_fine=function_space_fine,
            #             error_adapt_list=self.error_adapt_list,
            #             error_og_list=self.error_og_list,
            #             dur=dur_ms,
            #             sigma=self.sigma,
            #             alpha=self.alpha,
            #             r_0=self.r_0,
            #             t=self.t,
            #         )
            #     except fd.exceptions.ConvergenceError:
            #         fail_callback(self.t)
            #         print("Not Converged.")
            #         pass
            #     except Exception:
            #         print("fail!!!")
            #         raise Exception
            # time stepping and prep for next solving iter
            self.t += self.dt
            step += 1
            if step % 20 == 0:
                self.plot_fine_solution(step)
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
        function_space = fd.FunctionSpace(self.mesh, "CG", 1)
        u_og = fd.Function(function_space).project(self.u_cur)
        u_og_2_fine = fd.project(u_og, function_space_fine)

        # solve on coarse adapt mesh
        self.mesh.coordinates.dat.data[:] = self.adapt_coord
        self.mesh.coordinates.dat.data[:] = self.adapt_coord
        self.project_u_()
        self.solve_u(self.t)
        function_space_new = fd.FunctionSpace(self.mesh_new, "CG", 1)
        u_adapt = fd.Function(function_space_new).project(self.u_cur)
        u_adapt_2_fine = fd.project(u_adapt, function_space_fine)

        # error calculation
        error_og = fd.errornorm(u_fine, u_og_2_fine, norm_type="L2")
        error_adapt = fd.errornorm(u_fine, u_adapt_2_fine, norm_type="L2")

        # put mesh to init state
        self.mesh.coordinates.dat.data[:] = self.init_coord

        return error_og, error_adapt
    

    # def plot_fine_solution(self, index):
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     cmap = "seismic"
    #     # Solution on high resolution mesh
    #     cb = fd.tripcolor(self.u_cur_fine, cmap=cmap, axes=ax[0])
    #     plt.colorbar(cb)
    #     ax[0].set_title(f"u_cur_fine")
    #     cb = fd.tripcolor(self.u_fine, cmap=cmap, axes=ax[1])
    #     plt.colorbar(cb)
    #     ax[1].set_title(f"u_fine")
    #     plt.savefig(f"tmp_ret/u_fine_{index:04d}.png")
    #     plt.close()
    def plot_fine_solution(self, index):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cmap = "seismic"
        # Solution on high resolution mesh
        cb = fd.tripcolor(self.u_cur_fine, cmap=cmap, axes=ax)
        plt.colorbar(cb)
        ax.set_title(f"u_cur_fine")
        # cb = fd.tripcolor(self.u_fine, cmap=cmap, axes=ax[1])
        # plt.colorbar(cb)
        # ax[1].set_title(f"u_fine")
        plt.savefig(f"tmp_ret/u_fine_{index:04d}.png")
        plt.close()

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

        ax4 = fig.add_subplot(2, 3, 4, projection="3d")
        ax4.set_title("Hessian norm")
        fd.trisurf(self.f_norm, axes=ax4)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title("Orignal mesh")
        fd.tripcolor(self.uh, axes=ax5, cmap="coolwarm")
        fd.triplot(self.mesh, axes=ax5)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title("adapted mesh")
        fd.tripcolor(self.uh_new, axes=ax6, cmap="coolwarm")
        fd.triplot(self.mesh_new, axes=ax6)

        return fig


if __name__ == "__main__":
    n_grid = 20
    print("============== SwirlSolver =============/n")
    mesh = fd.UnitSquareMesh(n_grid, n_grid)
    mesh_new = fd.UnitSquareMesh(n_grid, n_grid)
    mesh_fine = fd.UnitSquareMesh(100, 100)
    swril_solver = SwirlSolver(mesh, mesh_fine, mesh_new, T=1, n_step=600)
    swril_solver.solve_problem()
