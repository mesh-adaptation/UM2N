# Author: Chunyang Wang
# GitHub Username: chunyang-w
import os
import firedrake as fd

os.environ['OMP_NUM_THREADS'] = "1"
__all__ = [
    "RandHelmholtzEqGenerator",
    "RandPossionEqGenerator",
    "HelmholtzEqGenerator",
]


# Generate a random Helmholtz equation based on a Gaussian distribution.
# The funnction has the form:
# $-\delta^{2} \mu + \mu = f$
class RandHelmholtzEqGenerator():
    def __init__(self, rand_u_generator):
        self.rand_u_generator = rand_u_generator
        self.problem_name = "rand_helmholtz"

    def discretise(self, mesh):
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
            })
        # Discretised Eq Definition Start
        self.f = -1 * fd.div(fd.grad(self.u_exact)) + self.u_exact
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = (fd.dot(
            fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(
            self.function_space, self.u_exact, "on_boundary")
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


class HelmholtzEqGenerator():
    def __init__(self, params={
        "f_func": None,
        "u_exact_func": None,
    }):
        self.f_func = params["f_func"]
        self.u_exact = params["u_exact_func"]

    def discretise(self, mesh):
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        self.function_space = V

        # Discretised Eq Definition Start
        self.f = self.f_func(x, y)
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = (fd.dot(
            fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(
            self.function_space, self.u_exact(x, y), "on_boundary")
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


class RandPossionEqGenerator():
    def __init__(self, rand_u_generator):
        self.rand_u_generator = rand_u_generator
        self.problem_name = "rand_helmholtz"

    def discretise(self, mesh):
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
            })
        # Discretised Eq Definition Start
        self.f = -1 * fd.div(fd.grad(self.u_exact))
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.LHS = (fd.dot(
            fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.bc = fd.DirichletBC(
            self.function_space, self.u_exact, "on_boundary")
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
