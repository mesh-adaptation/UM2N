# Author: Chunyang Wang, Mingrui Zhang
# GitHub Username: chunyang-w
# Description: Solve advection swirl problem

import time  # noqa
import os
import pickle
import firedrake as fd  # noqa
import torch
import movement as mv  # noqa
import UM2N
import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader
import firedrake.function as ffunc
import firedrake.functionspace as ffs
import ufl

import matplotlib.pyplot as plt  # noqa

from tqdm import tqdm  # noqa
from UM2N.model.train_util import model_forward


def get_log_og(log_path, idx):
    """
    Read log file from dataset log dir and return value in it
    """
    df = pd.read_csv(os.path.join(log_path, f"log{idx:04d}.csv"))
    return {
        "error_og": df["error_og"][0],
        "error_adapt": df["error_adapt"][0],
        "time": df["time"][0],
    }


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

    def __init__(self, mesh, mesh_fine, mesh_new, mesh_model, **kwargs):
        """
        Init the problem:
            1. define problem on fine mesh and coarse mesh
            2. init function space on fine & coarse mesh
            3. define hessian solver on coarse mesh
        """
        self.mesh = mesh  # coarse mesh
        self.mesh_fine = mesh_fine  # fine mesh
        self.mesh_new = mesh_new  # adapted mesh
        self.mesh_model = mesh_model  # model output mesh
        self.save_interval = kwargs.pop("save_interval", 5)
        self.n_monitor_smooth = kwargs.pop("n_monitor_smooth", 10)
        self.dataset = kwargs.pop(
            "dataset", None
        )  # dataset containing all data (eval use, set to None when generating data)
        # Init coords setup
        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = (
            self.mesh_fine.coordinates.vector().array().reshape(-1, 2)
        )  # noqa
        self.best_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.adapt_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        self.adapt_coord_prev = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
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
        self.tensor_space = fd.TensorFunctionSpace(self.mesh, "CG", 1)
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
        # self.dt = self.T / self.n_step
        self.dt = kwargs.pop("dt", 1e-3)
        self.dtc = fd.Constant(self.dt)
        # initial condition params
        self.sigma = kwargs.pop("sigma", (0.05 / 6))
        self.alpha = kwargs.pop("alpha", 1.5)
        self.r_0 = kwargs.pop("r_0", 0.2)
        self.x_0 = kwargs.pop("x_0", 0.25)
        self.y_0 = kwargs.pop("y_0", 0.25)

        # initital condition of u on coarse / fine mesh
        u_init_exp = get_u_0(self.x, self.y, self.r_0, self.x_0, self.y_0, self.sigma)  # noqa
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
        self.u_fine = fd.Function(self.scalar_space_fine).assign(self.u_init_fine)  # noqa
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

        # These two are holders for solution last timestep for coarse and adapted mesh
        self.u_prev_coarse = fd.Function(self.scalar_space).assign(self.u_init)

        self.mesh_prev = fd.Mesh(self.mesh.coordinates.copy(deepcopy=True))
        self.scalar_space_prev = fd.FunctionSpace(self.mesh_prev, "DG", 1)
        self.u_init_prev = fd.Function(self.scalar_space_prev).interpolate(u_init_exp)
        self.u_prev_adapt = fd.Function(self.scalar_space_prev).assign(self.u_init_prev)

        # Mesh holder for model
        self.mesh_model_prev = fd.Mesh(self.mesh.coordinates.copy(deepcopy=True))
        self.scalar_space_model_prev = fd.FunctionSpace(self.mesh_model_prev, "DG", 1)
        self.u_init_model_prev = fd.Function(self.scalar_space_model_prev).interpolate(
            u_init_exp
        )
        self.u_prev_model = fd.Function(self.scalar_space_model_prev).assign(
            self.u_init_model_prev
        )

        #       velocity field - the swirl: c
        self.c = fd.Function(self.vector_space)
        self.c_fine = fd.Function(self.vector_space_fine)
        self.cn = 0.5 * (fd.dot(self.c, self.n) + abs(fd.dot(self.c, self.n)))
        self.cn_fine = 0.5 * (
            fd.dot(self.c_fine, self.n_fine) + abs(fd.dot(self.c_fine, self.n_fine))
        )  # noqa

        # PDE problem RHS on coarse & fine mesh
        self.a = self.phi * self.du_trial * fd.dx(domain=self.mesh)
        self.a_fine = self.phi_fine * self.du_trial_fine * fd.dx(domain=self.mesh_fine)  # noqa

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
        self.solv1 = fd.LinearVariationalSolver(self.prob1, solver_parameters=params)  # noqa
        self.prob2 = fd.LinearVariationalProblem(self.a, self.L2, self.du)
        self.solv2 = fd.LinearVariationalSolver(self.prob2, solver_parameters=params)  # noqa
        self.prob3 = fd.LinearVariationalProblem(self.a, self.L3, self.du)
        self.solv3 = fd.LinearVariationalSolver(self.prob3, solver_parameters=params)  # noqa
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
        self.H, self.τ = (
            fd.TrialFunction(self.tensor_space),
            fd.TestFunction(self.tensor_space),
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

    def _compute_gradient_and_hessian(self, field, solver_parameters=None):
        mesh = self.tensor_space.mesh()
        V = ffs.VectorFunctionSpace(mesh, "CG", 1)
        W = V * self.tensor_space
        g, H = fd.TrialFunctions(W)
        phi, tau = fd.TestFunctions(W)
        sol = ffunc.Function(W)
        n = ufl.FacetNormal(mesh)

        a = (
            ufl.inner(tau, H) * ufl.dx
            + ufl.inner(ufl.div(tau), g) * ufl.dx
            - ufl.dot(g, ufl.dot(tau, n)) * ufl.ds
            - ufl.dot(ufl.avg(g), ufl.jump(tau, n)) * ufl.dS
            + ufl.inner(phi, g) * ufl.dx
        )
        L = (
            field * ufl.dot(phi, n) * ufl.ds
            + ufl.avg(field) * ufl.jump(phi, n) * ufl.dS
            - field * ufl.div(phi) * ufl.dx
        )
        if solver_parameters is None:
            solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "gmres",
                "ksp_max_it": 20,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_0_fields": "1",
                "pc_fieldsplit_1_fields": "0",
                "pc_fieldsplit_schur_precondition": "selfp",
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "gamg",
                "fieldsplit_1_mg_levels_ksp_max_it": 5,
            }
            if fd.COMM_WORLD.size == 1:
                solver_parameters["fieldsplit_0_pc_type"] = "ilu"
                solver_parameters["fieldsplit_1_mg_levels_pc_type"] = "ilu"
            else:
                solver_parameters["fieldsplit_0_pc_type"] = "bjacobi"
                solver_parameters["fieldsplit_0_sub_ksp_type"] = "preonly"
                solver_parameters["fieldsplit_0_sub_pc_type"] = "ilu"
                solver_parameters["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
                solver_parameters["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
                solver_parameters["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"
        fd.solve(a == L, sol, solver_parameters=solver_parameters)
        return sol.subfunctions

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
        self.u2_fine.assign(0.75 * self.u_fine + 0.25 * (self.u1_fine + self.du_fine))  # noqa

        self.solv3_fine.solve()
        self.u_cur_fine.assign(
            (1.0 / 3.0) * self.u_fine + (2.0 / 3.0) * (self.u2_fine + self.du_fine)
        )  # noqa

    def project_u_(self):
        self.u.project(self.u_fine_buffer)
        return

    def project_from_prev_u_coarse(self):
        self.u.project(self.u_prev_coarse)
        return

    def project_from_prev_u_adapt(self):
        self.u.project(self.u_prev_adapt)
        return

    def project_from_prev_u_model(self):
        self.u.project(self.u_prev_model)
        return

    def monitor_function(self, mesh, alpha=10, beta=5):
        # self.project_u_()
        self.project_from_prev_u_adapt()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.l2_projection = self._compute_gradient_and_hessian(self.u_hess)[1]

        # self.hessian_prob.solve()
        self.f_norm.interpolate(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.interpolate(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # filter_monitor_val = np.minimum(1e3, self.f_norm.dat.data[:])
        # self.f_norm.dat.data[:] = filter_monitor_val

        print(
            f"max and min monitors Hessian {self.f_norm.dat.data[:].max()}, {self.f_norm.dat.data[:].min()}"
        )
        print(
            f"max and min grad norm {self.grad_norm.dat.data[:].max()}, {self.grad_norm.dat.data[:].min()}"
        )
        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()

        monitor_values_dg = fd.Function(fd.FunctionSpace(mesh, "DG", 1))
        monitor_values = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        # Choose the max values between grad norm and hessian norm according to
        # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
        monitor_values_dg.dat.data[:] = np.maximum(
            beta * self.f_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
        )
        monitor_values.project(monitor_values_dg)

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

        self.adapt_coord = mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        monitor_values.project(1 + monitor_smoothed)

        return monitor_values

    def monitor_function_on_coarse_mesh(self, mesh, alpha=10, beta=5):
        # self.project_u_()
        self.project_from_prev_u_adapt()
        self.solve_u(self.t)
        self.u_hess.project(self.u_cur)

        self.l2_projection = self._compute_gradient_and_hessian(self.u_hess)[1]

        # self.hessian_prob.solve()
        self.f_norm.interpolate(
            self.l2_projection[0, 0] ** 2
            + self.l2_projection[0, 1] ** 2
            + self.l2_projection[1, 0] ** 2
            + self.l2_projection[1, 1] ** 2
        )

        func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.u_cur), func_vec_space)
        self.grad_norm.interpolate(uh_grad[0] ** 2 + uh_grad[1] ** 2)

        # Normlize the hessian
        self.f_norm /= self.f_norm.vector().max()
        # Normlize the grad
        self.grad_norm /= self.grad_norm.vector().max()

        monitor_values_dg = fd.Function(fd.FunctionSpace(mesh, "DG", 1))
        monitor_values = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        # Choose the max values between grad norm and hessian norm according to
        # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
        monitor_values_dg.dat.data[:] = np.maximum(
            beta * self.f_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
        )
        monitor_values.project(monitor_values_dg)

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
        adapter = mv.MongeAmpereMover(
            self.mesh, monitor_function=self.monitor_function, rtol=1e-3, maxiter=500
        )
        for i in range(self.n_step):
            print(f"step: {step}, t: {self.t:.5f}")
            # error tracking lists init
            self.error_adapt_list = []
            self.error_og_list = []
            # solve PDE problem on fine mesh
            self.solve_u_fine(self.t)
            # if ((step + 1) % self.save_interval == 0) or (step == 0):
            print(f"---- getting samples: step: {step}, t: {self.t:.5f}")
            try:
                monitor_values = self.monitor_function_on_coarse_mesh(self.mesh)
                # Record the hessian norm on coarse mesh
                function_scalar_space_cg = fd.FunctionSpace(self.mesh, "CG", 1)
                grad_u_norm_cg = fd.Function(function_scalar_space_cg).project(
                    self.grad_norm
                )
                hessian_norm_cg = fd.Function(function_scalar_space_cg).project(
                    self.f_norm
                )

                # # TODO: Starting the mesh movement from last adapted mesh (This is not working currently)
                # self.mesh.coordinates.dat.data[:] = self.adapt_coord_prev
                # mesh movement - calculate the adapted coords
                start = time.perf_counter()
                # adapter = mv.MongeAmpereMover(
                #     self.mesh, monitor_function=self.monitor_function, rtol=1e-3, maxiter=100
                # )
                adapter.move()
                end = time.perf_counter()
                dur_ms = (end - start) * 1e3
                # self.mesh_new.coordinates.dat.data[:] = self.adapt_coord
                self.mesh_new.coordinates.dat.data[:] = (
                    adapter.mesh.coordinates.dat.data[:]
                )
                # self.adapt_coord_prev = self.mesh_new.coordinates.dat.data[:]

                # calculate solution on original mesh
                self.mesh.coordinates.dat.data[:] = self.init_coord
                self.project_from_prev_u_coarse()
                self.solve_u(self.t)
                function_space = fd.FunctionSpace(self.mesh, "CG", 1)
                uh = fd.Function(function_space).project(self.u_cur)
                # Update the prev solution on coarse mesh
                self.u_prev_coarse.project(self.u_cur)

                # calculate solution on adapted mesh
                self.mesh.coordinates.dat.data[:] = self.adapt_coord
                self.project_from_prev_u_adapt()
                self.solve_u(self.t)

                self.mesh_prev.coordinates.dat.data[:] = self.adapt_coord
                function_space_new = fd.FunctionSpace(self.mesh_prev, "CG", 1)  # noqa
                uh_new = fd.Function(function_space_new).project(self.u_cur)  # noqa
                # Update the prev solution on adapted mesh
                self.u_prev_adapt.project(self.u_cur)

                # Get the u_fine at current step
                function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)  # noqa
                uh_fine = fd.Function(function_space_fine)
                uh_fine.project(self.u_cur_fine)

                # Put mesh to init state
                self.mesh.coordinates.dat.data[:] = self.init_coord

                # error measure
                # Project to fine mesh
                # Note: this should only be performed after the "self.mesh" has been recovered back to initial uniform mesh
                u_og_2_fine = fd.project(uh, function_space_fine)
                u_adapt_2_fine = fd.project(uh_new, function_space_fine)

                error_og = fd.errornorm(u_og_2_fine, uh_fine, norm_type="L2")
                error_adapt = fd.errornorm(u_adapt_2_fine, uh_fine, norm_type="L2")
                print(
                    f"[Error measure] error_og: {error_og}, \terror_adapt: {error_adapt}"
                )

                func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
                uh_grad = fd.interpolate(fd.grad(uh), func_vec_space)

                hessian = self.l2_projection
                phi = adapter.phi
                phi_grad = adapter.grad_phi
                sigma = adapter.sigma
                I = fd.Identity(2)  # noqa
                jacobian = I + sigma
                jacobian_det = fd.Function(function_space, name="jacobian_det")
                jacobian_det.project(
                    jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
                )
                self.jacob_det = fd.project(
                    jacobian_det, fd.FunctionSpace(self.mesh, "CG", 1)
                )
                self.jacob = fd.project(
                    jacobian, fd.TensorFunctionSpace(self.mesh, "CG", 1)
                )

                if ((step + 1) % self.save_interval == 0) or (step == 0):
                    callback(
                        uh=uh,
                        uh_grad=uh_grad,
                        grad_u_norm=grad_u_norm_cg,
                        hessian_norm=hessian_norm_cg,
                        monitor_values=monitor_values,
                        hessian=hessian,
                        phi=phi,
                        grad_phi=phi_grad,
                        jacobian=self.jacob,
                        jacobian_det=self.jacob_det,
                        mesh_new=self.mesh_new,
                        mesh_og=self.mesh,
                        uh_new=uh_new,
                        uh_fine=uh_fine,
                        function_space=function_space,
                        function_space_fine=function_space_fine,
                        error_adapt_list=self.error_adapt_list,
                        error_og_list=self.error_og_list,
                        dur=dur_ms,
                        sigma=self.sigma,
                        alpha=self.alpha,
                        r_0=self.r_0,
                        t=self.t,
                    )

            except fd.exceptions.ConvergenceError:
                fail_callback(self.t)
                print("Not Converged.")
                pass
            except Exception:
                print("fail!!!")
                raise Exception
            # time stepping and prep for next solving iter
            self.t += self.dt
            step += 1
            self.u_fine.assign(self.u_cur_fine)
            self.u_fine_buffer.assign(self.u_cur_fine)

        return

    def eval_problem(
        self,
        model,
        ds_root,
        eval_dir,
        model_name="model",
        callback=None,
        fail_callback=None,
        device="cuda",
    ):
        print("Evaluating problem")
        log_path = os.path.join(eval_dir, "log")
        plot_path = os.path.join(eval_dir, "plot")
        plot_more_path = os.path.join(eval_dir, "plot_more")
        plot_data_path = os.path.join(eval_dir, "plot_data")
        self.make_all_dirs(log_path, plot_path, plot_more_path, plot_data_path)
        self.t = 0.0
        step = 0
        idx = 0
        res = {
            "deform_loss": None,  # nodal position loss
            "tangled_element": None,  # tangled elements on a mesh  # noqa
            "error_og": None,  # PDE error on original uniform mesh  # noqa
            "error_model": None,  # PDE error on model generated mesh   # noqa
            "error_ma": None,  # PDE error on MA generated mesh      # noqa
            "error_reduction_MA": None,  # PDE error reduced by using MA mesh  # noqa
            "error_reduction_model": None,  # PDE error reduced by using model mesh  # noqa
            "time_consumption_model": None,  # time consumed generating mesh inferenced by the model  # noqa
            "time_consumption_MA": None,  # time consumed generating mesh by Monge-Ampere method  # noqa
            "acceration_ratio": None,  # time_consumption_ma / time_consumption_model  # noqa
        }
        model.eval()
        model = model.to(device)
        for i in range(self.n_step):
            print(f"step: {step}, t: {self.t:.5f}")
            # data loading from raw file
            raw_data_path = self.dataset.file_names[idx]
            raw_data = np.load(raw_data_path, allow_pickle=True).item()
            y = raw_data.get("y")
            # print("raw data ", raw_data.keys())
            # error tracking lists init
            self.error_adapt_list = []
            self.error_og_list = []
            sample = next(
                iter(DataLoader([self.dataset[idx]], batch_size=1, shuffle=False))
            )

            # solve PDE problem on fine mesh
            self.solve_u_fine(self.t)
            # if ((step + 1) % self.save_interval == 0) or (step == 0):
            print(f"---- getting samples: step: {step}, t: {self.t:.5f}")

            start = time.perf_counter()
            bs = 1
            sample = sample.to(device)
            with torch.no_grad():
                start = time.perf_counter()
                if model_name == "MRTransformer" or model_name == "M2T":
                    data = sample
                    (
                        output_coord,
                        output,
                        out_monitor,
                        phix,
                        phiy,
                        mesh_query_x_all,
                        mesh_query_y_all,
                    ) = model_forward(
                        bs,
                        data,
                        model,
                        use_add_random_query=False,
                    )
                    out = output_coord
                elif model_name == "M2N":
                    out = model(sample)
                elif model_name == "MRN":
                    out = model(sample)
                elif model_name == "M2N_T":
                    out = model(sample)
                else:
                    raise Exception(f"model {model_name} not implemented.")
            end = time.perf_counter()
            dur_ms = (end - start) * 1e3

            # calculate solution on original mesh
            self.mesh.coordinates.dat.data[:] = self.init_coord  # For solve use
            self.project_from_prev_u_coarse()
            self.solve_u(self.t)
            function_space = fd.FunctionSpace(self.mesh, "CG", 1)
            uh = fd.Function(function_space).project(self.u_cur)
            # Update the prev solution on coarse mesh
            self.u_prev_coarse.project(self.u_cur)

            # # calculate solution on adapted mesh
            # self.mesh.coordinates.dat.data[:] = y
            # self.project_from_prev_u_adapt()
            # self.solve_u(self.t)

            # self.mesh_prev.coordinates.dat.data[:] = self.adapt_coord
            # function_space_new = fd.FunctionSpace(
            #     self.mesh_prev, "CG", 1
            # )  # noqa
            # uh_new = fd.Function(function_space_new).project(
            #     self.u_cur
            # )  # noqa
            # # Update the prev solution on adapted mesh
            # self.u_prev_adapt.project(self.u_cur)

            # check mesh integrity - Only perform evaluation on non-tangling mesh  # noqa
            num_tangle = UM2N.get_sample_tangle(out, sample.x[:, :2], sample.face)  # noqa
            if isinstance(num_tangle, torch.Tensor):
                num_tangle = num_tangle.item()
            if num_tangle > 0:  # has tangled elems:
                res["tangled_element"] = num_tangle
                res["error_model"] = -1
            else:  # mesh is valid, perform evaluation: 1.
                res["tangled_element"] = num_tangle

            # calculate solution on model output mesh
            self.mesh_model.coordinates.dat.data[:] = (
                out.detach().cpu().numpy()
            )  # For output use
            self.mesh.coordinates.dat.data[:] = out.detach().cpu().numpy()
            self.project_from_prev_u_model()
            self.solve_u(self.t)
            self.mesh_model_prev.coordinates.dat.data[:] = out.detach().cpu().numpy()
            function_space_new = fd.FunctionSpace(self.mesh_model_prev, "CG", 1)  # noqa
            uh_model = fd.Function(function_space_new).project(self.u_cur)  # noqa
            # Update the prev solution on model ouput mesh
            self.u_prev_model.project(self.u_cur)

            # Get the u_fine at current step
            function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)  # noqa
            uh_fine = fd.Function(function_space_fine)
            uh_fine.project(self.u_cur_fine)

            # Put mesh to init state
            self.mesh.coordinates.dat.data[:] = self.init_coord

            self.mesh_new.coordinates.dat.data[:] = y

            if ((step + 1) % self.save_interval == 0) or (step == 0):
                fig, plot_data_dict = UM2N.plot_compare(
                    self.mesh_fine,
                    self.mesh,
                    self.mesh_new,
                    self.mesh_model,
                    uh_fine,
                    uh,
                    # uh_new,
                    uh,
                    uh_model,
                    raw_data.get("hessian_norm")[:, 0],
                    raw_data.get("monitor_val")[:, 0],
                    num_tangle,
                    model_name,
                )
                res["deform_loss"] = 1000 * torch.nn.L1Loss()(out, sample.y).item()
                plot_data_dict["deform_loss"] = res["deform_loss"]

                fig.savefig(os.path.join(self.plot_more_path, f"plot_{idx:04d}.png"))  # noqa
                plt.close(fig)

                # Save plot data
                with open(
                    os.path.join(self.plot_data_path, f"plot_data_{idx:04d}.pkl"), "wb"
                ) as p:
                    pickle.dump(plot_data_dict, p)

                # Record the error
                error_model = plot_data_dict["error_norm_model"]
                error_og = plot_data_dict["error_norm_original"]
                error_ma = plot_data_dict["error_norm_ma"]

                print(f"error_og: {error_og}, \terror_ma: {error_ma}")

                res["error_og"] = error_og
                res["error_ma"] = error_ma
                res["error_model"] = error_model

                print("inspect out type: ", type(out.detach().cpu().numpy()))

                # get time_MA by reading log file
                res["time_consumption_MA"] = get_log_og(
                    os.path.join(ds_root, "log"), idx
                )["time"]
                print(res)

                # metric calculation
                # res["deform_loss"] = 1000 * torch.nn.L1Loss()(out, sample.y).item()
                res["time_consumption_model"] = dur_ms

                res["acceration_ratio"] = (
                    res["time_consumption_MA"] / res["time_consumption_model"]
                )  # noqa
                res["error_reduction_MA"] = (res["error_og"] - res["error_ma"]) / res[
                    "error_og"
                ]  # noqa
                res["error_reduction_model"] = (
                    res["error_og"] - res["error_model"]
                ) / res["error_og"]  # noqa

                # save file
                df = pd.DataFrame(res, index=[0])
                df.to_csv(os.path.join(self.log_path, f"log_{idx:04d}.csv"))

                idx += 1

            # time stepping and prep for next solving iter
            self.t += self.dt
            step += 1
            self.u_fine.assign(self.u_cur_fine)
            self.u_fine_buffer.assign(self.u_cur_fine)

    def make_all_dirs(self, log_path, plot_path, plot_more_path, plot_data_path):
        self.log_path = log_path
        self.plot_path = plot_path
        self.plot_more_path = plot_more_path
        self.plot_data_path = plot_data_path

        self.make_log_dir(log_path)
        self.make_plot_dir(plot_path)
        self.make_plot_more_dir(plot_more_path)
        self.make_plot_data_dir(plot_data_path)

    def make_log_dir(self, log_path):
        UM2N.mkdir_if_not_exist(log_path)

    def make_plot_dir(self, plot_path):
        UM2N.mkdir_if_not_exist(plot_path)

    def make_plot_more_dir(self, plot_more_path):
        UM2N.mkdir_if_not_exist(plot_more_path)

    def make_plot_data_dir(self, plot_data_path):
        UM2N.mkdir_if_not_exist(plot_data_path)


if __name__ == "__main__":
    n_grid = 20
    print("============== SwirlSolver =============/n")
    mesh = fd.UnitSquareMesh(n_grid, n_grid)
    mesh_new = fd.UnitSquareMesh(n_grid, n_grid)
    mesh_fine = fd.UnitSquareMesh(100, 100)
    swril_solver = SwirlSolver(mesh, mesh_fine, mesh_new, T=1, n_step=600)
    swril_solver.solve_problem()
