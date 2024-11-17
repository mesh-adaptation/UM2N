import firedrake as fd
import numpy as np
import pytest
import ufl

from UM2N.generator.equation_generator import HelmholtzEqGenerator, PoissonEqGenerator


@pytest.fixture(params=["poisson", "helmholtz"])
def equation(request):
    return request.param


generators = {
    "poisson": PoissonEqGenerator,
    "helmholtz": HelmholtzEqGenerator,
}


def linear_expr(x, y):
    return x


def test_linear(equation):
    """
    Test that a Poisson or Helmholtz equation with a linear exact solution can be solved
    to high accuracy.
    """
    generator = generators[equation](params={"u_exact_func": linear_expr})
    mesh = fd.UnitSquareMesh(4, 4)
    ret = generator.discretise(mesh)
    assert ret["mesh"] is mesh
    u = fd.Function(ret["function_space"])
    fd.solve(ret["LHS"] == ret["RHS"], u, bcs=ret["bc"])
    x, y = ufl.SpatialCoordinate(mesh)
    assert np.isclose(fd.assemble(u * ufl.dx), fd.assemble(linear_expr(x, y) * ufl.dx))
