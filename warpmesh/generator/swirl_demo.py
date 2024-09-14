# Author: Chunyang Wang
# GitHub Username: chunyang-w
# Description: A script to solve advection swirl problem in Fig 19.
# Link to original paper: https://www.sciencedirect.com/science/article/abs/pii/S002199912300476X?casa_token=uw9QIN0ceC8AAAAA:wr7Y3n_pKe_TUdaR-6VlTti3-SSPWc0Nelwnks5Kv6hkfWMtqbypYk_XN8DtPIAhaBGD8LoUjw # noqa
# key take aways from the paper:
#     1. Use discontinuous Galerkin (DG) FEM methods

import firedrake as fd

from matplotlib import pyplot as plt

n_grid = 60
T = 1
n_step = 600
dt = T / n_step

sigma = 0.05 / 6
x_0 = 0.25
y_0 = 0.25
r_0 = 0.2


def get_c(x, y, t, threshold=0.5):
    """
    Compute the velocity field which transports the
    solution field u.

    Return:
        velocity (ufl.tensors): expression of the swirl velocity field
    """
    a = 1 if t < threshold else -1
    v_x = (3 / 2) * a * fd.sin(fd.pi * x) ** 2 * fd.sin(2 * fd.pi * y)
    v_y = -1 * (3 / 2) * a * fd.sin(fd.pi * y) ** 2 * fd.sin(2 * fd.pi * x)
    velocity = fd.as_vector((v_x, v_y))
    return velocity


def get_u_0(x, y, r_0=0.2, x_0=0.25, y_0=0.25, sigma=(0.05 / 3)):
    """
    Compute the initial trace value.

    Return:
        u_0 (ufl.tensors): expression of u_0
    """
    u = fd.exp(
        (-1 / (2 * sigma)) * (fd.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) - r_0) ** 2
    )
    return u


if __name__ == "__main__":
    mesh = fd.UnitSquareMesh(n_grid, n_grid)
    vector_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    scalar_space = fd.FunctionSpace(mesh, "CG", 1)

    du_trial = fd.TrialFunction(scalar_space)
    phi = fd.TestFunction(scalar_space)
    n = fd.FacetNormal(mesh)

    x, y = fd.SpatialCoordinate(mesh)
    dtc = fd.Constant(dt)

    step = 0
    t = 0.0

    u_0_exp = get_u_0(x, y, r_0, x_0, y_0, sigma)
    u_0 = fd.Function(scalar_space).interpolate(u_0_exp)
    u_in = fd.Constant(0.0)
    u = fd.Function(scalar_space).assign(u_0)
    u1 = fd.Function(scalar_space)
    u2 = fd.Function(scalar_space)

    c_exp = get_c(x, y, t)
    c = fd.Function(vector_space).interpolate(c_exp)
    cn = 0.5 * (fd.dot(c, n) + abs(fd.dot(c, n)))

    a = phi * du_trial * fd.dx

    L1 = dtc * (
        u * fd.div(phi * c) * fd.dx
        - fd.conditional(fd.dot(c, n) < 0, phi * fd.dot(c, n) * u_in, 0.0) * fd.ds  # noqa
        - fd.conditional(fd.dot(c, n) > 0, phi * fd.dot(c, n) * u, 0.0) * fd.ds
        - (phi("+") - phi("-")) * (cn("+") * u("+") - cn("-") * u("-")) * fd.dS
    )
    L2 = fd.replace(L1, {u: u1})
    L3 = fd.replace(L1, {u: u2})

    du = fd.Function(scalar_space)

    params = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}  # noqa
    prob1 = fd.LinearVariationalProblem(a, L1, du)
    solv1 = fd.LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = fd.LinearVariationalProblem(a, L2, du)
    solv2 = fd.LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = fd.LinearVariationalProblem(a, L3, du)
    solv3 = fd.LinearVariationalSolver(prob3, solver_parameters=params)

    for i in range(n_step):
        print(t)
        c_exp = get_c(x, y, t)
        c_temp = fd.Function(vector_space).interpolate(c_exp)
        c.project(c_temp)

        solv1.solve()
        u1.assign(u + du)

        solv2.solve()
        u2.assign(0.75 * u + 0.25 * (u1 + du))

        solv3.solve()
        u.assign((1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + du))

        t += dt
        step += 1
        if step % 20 == 0:
            fd.tripcolor(u)
    plt.show()
