"""
Module for generating random equations.
"""

import abc

import firedrake as fd

__all__ = [
    "RandHelmholtzEqGenerator",
    "RandPoissonEqGenerator",
    "HelmholtzEqGenerator",
    "PoissonEqGenerator",
]


class EquationGenerator(abc.ABC):
    """
    Base class for equation generators.
    """


class RandomEquationGenerator(abc.ABC):
    """
    Base class for random equation generators.
    """


class RandHelmholtzEqGenerator(RandomEquationGenerator):
    r"""
      Generate a random Helmholtz equation based on a Gaussian distribution.

      The function has the form:

    ..math::
          -\delta^{2} \mu + \mu = f
    """

    def __init__(self, rand_u_generator):
        # TODO: Docstring
        self.rand_u_generator = rand_u_generator
        self.problem_name = "rand_helmholtz"

    def discretise(self, mesh):
        # TODO: Docstring; use HelmholtzEqGenerator code
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        self.function_space = V
        # use generator to generate u_exact
        self.u_exact = self.rand_u_generator.get_u_exact(
            params={
                "x": x,
                "y": y,
            }
        )
        # Discretised Eq Definition Start
        self.f = -1 * fd.div(fd.grad(self.u_exact)) + self.u_exact
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = (fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(self.function_space, self.u_exact, "on_boundary")
        # Discretised Eq Definition End
        return {
            "mesh": mesh,
            "function_space": self.function_space,
            "u_exact": self.u_exact,
            "LHS": self.LHS,
            "RHS": self.RHS,
            "bc": self.bc,
            "f": self.f,
        }


class HelmholtzEqGenerator(EquationGenerator):
    def __init__(
        self,
        params={
            "u_exact_func": None,
        },
    ):
        self.u_exact_func = params["u_exact_func"]

    def discretise(self, mesh):
        # TODO: Docstring
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        self.function_space = V
        self.u_exact = self.u_exact_func(x, y)

        # Discretised Eq Definition Start
        self.f = -1 * fd.div(fd.grad(self.u_exact)) + self.u_exact
        # self.f = self.f_func(x, y)
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = (fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(self.function_space, self.u_exact, "on_boundary")
        # Discretised Eq Definition End
        return {
            "mesh": mesh,
            "function_space": self.function_space,
            "u_exact": self.u_exact,
            "LHS": self.LHS,
            "RHS": self.RHS,
            "bc": self.bc,
            "f": self.f,
        }


class RandPoissonEqGenerator(RandomEquationGenerator):
    # TODO: Docstring
    def __init__(self, rand_u_generator):
        # TODO: Docstring
        self.rand_u_generator = rand_u_generator
        self.problem_name = "rand_helmholtz"

    def discretise(self, mesh):
        # TODO: Docstring; use PoissonEqGenerator code
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        self.function_space = V
        # use generator to generate u_exact
        self.u_exact = self.rand_u_generator.get_u_exact(
            params={
                "x": x,
                "y": y,
            }
        )
        # Discretised Eq Definition Start
        self.f = -1 * fd.div(fd.grad(self.u_exact))
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = fd.dot(fd.grad(v), fd.grad(u)) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(self.function_space, self.u_exact, "on_boundary")
        # Discretised Eq Definition End
        return {
            "mesh": mesh,
            "function_space": self.function_space,
            "u_exact": self.u_exact,
            "LHS": self.LHS,
            "RHS": self.RHS,
            "bc": self.bc,
            "f": self.f,
        }


class PoissonEqGenerator(EquationGenerator):
    # TODO: Docstring
    def __init__(
        # TODO: Docstring
        self,
        params={
            "u_exact_func": None,
        },
    ):
        self.u_exact_func = params["u_exact_func"]

    def discretise(self, mesh):
        # TODO: Docstring
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        self.function_space = V
        self.u_exact = self.u_exact_func(x, y)

        # Discretised Eq Definition Start
        self.f = -1 * fd.div(fd.grad(self.u_exact))
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = fd.dot(fd.grad(v), fd.grad(u)) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(self.function_space, self.u_exact, "on_boundary")
        # Discretised Eq Definition End
        return {
            "mesh": mesh,
            "function_space": self.function_space,
            "u_exact": self.u_exact,
            "LHS": self.LHS,
            "RHS": self.RHS,
            "bc": self.bc,
            "f": self.f,
        }
