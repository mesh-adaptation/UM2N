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

        monitor_val = 1 + 5 * hessian_norm
        self.monitor_val = fd.Function(function_space)
        self.monitor_val.assign(monitor_val)
        return monitor_val
