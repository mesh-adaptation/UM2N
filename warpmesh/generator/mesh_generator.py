import os
import numpy as np
import firedrake as fd
import movement as mv
from warpmesh.generator.equation_solver import EquationSolver

os.environ["OMP_NUM_THREADS"] = "1"
__all__ = ["MeshGenerator"]


class MeshGenerator:
    """
    Responsible for generating and moving a mesh based on a given Helmholtz
     equation.
    This method is based on Pyroteus/movement.

    Attributes:
    - eq: The Helmholtz equation object.
    - num_grid_x: Number of grid points in the x-dimension.
    - num_grid_y: Number of grid points in the y-dimension.
    - mesh: The initial m
    esh.
    """

    def __init__(
        self,
        params={
            "num_grid_x": None,
            "num_grid_y": None,
            "eq": None,
            "mesh": None,
        },
    ):
        self.eq = params["eq"]
        # self.num_grid_x = params["num_grid_x"]
        # self.num_grid_y = params["num_grid_y"]
        self.mesh = params["mesh"]

        self.monitor_val = None
        self.uh = None
        self.n_monitor_smooth = 15

    def move_mesh(self):
        """
        Moves the mesh using the Monge-Ampere equation.
        Computes and stores the Jacobian and its determinant.

        Returns:
        - The moved mesh
        """
        mover = mv.MongeAmpereMover(
            self.mesh, self.monitor_func, method="relaxation", rtol=1e-3, maxiter=500
        )
        mover.move()
        # extract Hessian of the movement
        sigma = mover.sigma
        I = fd.Identity(2)  # noqa
        jacobian = I + sigma
        jacobian_det = fd.Function(self.eq.function_space, name="jacobian_det")
        jacobian_det.project(
            jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
        )
        self.jacob_det = jacobian_det
        self.jacob = jacobian
        # extract phi of the movement
        self.phi = mover.phi
        # extract phi_grad
        self.grad_phi = mover.grad_phi

        return self.mesh

    def get_grad_phi(self):
        """
        Returns the gradient of phi of the mesh movement.
        """
        return self.grad_phi

    def get_phi(self):
        """
        Returns the phi of the mesh movement.
        """
        return self.phi

    def get_monitor_val(self):
        """
        Returns the monitor function value used for mesh movement.
        """
        return self.monitor_val

    def get_jacobian(self):
        """
        Returns the Jacobian of the mesh movement.
        """
        return self.jacob

    def get_jacobian_det(self):
        """
        Returns the determinant of the Jacobian of the mesh movement.
        """
        return self.jacob_det

    def get_gradient(self, mesh):
        res = self.eq.discretise(mesh)
        # function_space_ten = fd.TensorFunctionSpace(mesh, "CG", 1)

        solver = EquationSolver(
            params={
                "LHS": res["LHS"],
                "RHS": res["RHS"],
                "function_space": res["function_space"],
                "bc": res["bc"],
            }
        )
        uh = solver.solve_eq()
        self.uh = uh
        func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(self.uh), func_vec_space)
        return uh_grad

    def get_grad_norm(self, mesh):
        res = self.eq.discretise(mesh)
        uh_grad = self.get_gradient(mesh)

        grad_norm = fd.Function(res["function_space"])
        grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)
        grad_norm /= grad_norm.vector().max()

        return grad_norm

    def get_hessian(self, mesh):
        """
        Computes and returns the Hessian of the Helmholtz equation on the
        given mesh.

        Parameters:
        - mesh: The mesh on which to compute the Hessian.

        Returns:
        - The Hessian as a projection in the function space.
        """
        res = self.eq.discretise(mesh)
        function_space_ten = fd.TensorFunctionSpace(mesh, "CG", 1)

        solver = EquationSolver(
            params={
                "LHS": res["LHS"],
                "RHS": res["RHS"],
                "function_space": res["function_space"],
                "bc": res["bc"],
            }
        )
        uh = solver.solve_eq()
        self.uh = uh

        n = fd.FacetNormal(mesh)
        l2_projection = fd.Function(function_space_ten)
        H, h = fd.TrialFunction(function_space_ten), fd.TestFunction(function_space_ten)
        a = fd.inner(h, H) * fd.dx(domain=mesh)
        L = -fd.inner(fd.div(h), fd.grad(uh)) * fd.dx(domain=mesh)
        L += fd.dot(fd.grad(uh), fd.dot(h, n)) * fd.ds(domain=mesh)
        prob = fd.LinearVariationalProblem(a, L, l2_projection)
        hessian_prob = fd.LinearVariationalSolver(prob)
        hessian_prob.solve()
        return l2_projection

    def get_hessian_norm(self, mesh):
        res = self.eq.discretise(mesh)
        function_space = res["function_space"]
        hessian_norm = fd.Function(function_space)
        l2_projection = self.get_hessian(mesh)
        hessian_norm.project(
            l2_projection[0, 0] ** 2
            + l2_projection[0, 1] ** 2
            + l2_projection[1, 0] ** 2
            + l2_projection[1, 1] ** 2
        )
        hessian_norm /= hessian_norm.vector().max()
        return hessian_norm

    def monitor_func(self, mesh):
        """
        Computes the monitor function value based on the Hessian of the
        Helmholtz equation.

        Parameters:
        - mesh: The mesh on which to compute the monitor function.

        Returns:
        - The monitor function value.
        """
        res = self.eq.discretise(mesh)
        function_space = res["function_space"]
        hessian_norm = fd.Function(function_space)
        l2_projection = self.get_hessian(mesh)
        hessian_norm.project(
            l2_projection[0, 0] ** 2
            + l2_projection[0, 1] ** 2
            + l2_projection[1, 0] ** 2
            + l2_projection[1, 1] ** 2
        )
        hessian_norm /= hessian_norm.vector().max()

        raw_monitor_val = 1 + 5 * hessian_norm
        monitor_val = fd.Function(function_space)
        monitor_val.assign(raw_monitor_val)
        return monitor_val

    # def monitor_func(self, mesh, alpha=10, beta=5):
    #     # self.project_u_()
    #     # self.solve_u(self.t)
    #     # self.u_hess.project(self.u_cur)

    #     # self.hessian_prob.solve()
    #     # self.f_norm.project(
    #     #     self.l2_projection[0, 0] ** 2
    #     #     + self.l2_projection[0, 1] ** 2
    #     #     + self.l2_projection[1, 0] ** 2
    #     #     + self.l2_projection[1, 1] ** 2
    #     # )

    #     res = self.eq.discretise(mesh)
    #     function_space = res["function_space"]
    #     hessian_norm = fd.Function(function_space)
    #     l2_projection = self.get_hessian(mesh)
    #     hessian_norm.project(
    #         l2_projection[0, 0] ** 2
    #         + l2_projection[0, 1] ** 2
    #         + l2_projection[1, 0] ** 2
    #         + l2_projection[1, 1] ** 2
    #     )

    #     func_vec_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
    #     uh_grad = fd.interpolate(fd.grad(self.uh), func_vec_space)

    #     self.grad_norm = fd.Function(function_space)
    #     self.grad_norm.project(uh_grad[0] ** 2 + uh_grad[1] ** 2)

    #     # Normlize the hessian
    #     self.hessian_norm /= self.hessian_norm.vector().max()
    #     # Normlize the grad
    #     self.grad_norm /= self.grad_norm.vector().max()

    #     self.monitor_val = fd.Function(function_space)
    #     # Choose the max values between grad norm and hessian norm according to
    #     # [Clare et al 2020] Multi-scale hydro-morphodynamic modelling using mesh movement methods
    #     self.monitor_val.dat.data[:] = np.maximum(
    #         beta * self.hessian_norm.dat.data[:], alpha * self.grad_norm.dat.data[:]
    #     )

    #     # #################

    #     V = fd.FunctionSpace(mesh, "CG", 1)
    #     u = fd.TrialFunction(V)
    #     v = fd.TestFunction(V)
    #     function_space = V
    #     # Discretised Eq Definition Start
    #     f = self.monitor_val
    #     dx = mesh.cell_sizes.dat.data[:].mean()
    #     N = self.n_monitor_smooth
    #     K = N * dx**2 / 4
    #     RHS = f * v * fd.dx(domain=mesh)
    #     LHS = (K * fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
    #     bc = fd.DirichletBC(function_space, f, "on_boundary")

    #     monitor_smoothed = fd.Function(function_space)
    #     fd.solve(
    #         LHS == RHS,
    #         monitor_smoothed,
    #         solver_parameters={"ksp_type": "cg", "pc_type": "none"},
    #         bcs=bc,
    #     )

    #     # #################
    #     monitor_val = 1 + monitor_smoothed
    #     self.monitor_val.assign(monitor_val)
    #     return monitor_val
