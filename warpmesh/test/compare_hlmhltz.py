# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import firedrake as fd
import warpmesh as wm
import matplotlib.pyplot as plt # noqa

__all__ = ['compare_error']


def compare_error(data_in,
                  mesh, mesh_fine, mesh_model, mesh_MA, num_tangle,
                  problem_type='helmholtz'
                  ):
    # read in params used to construct the analytical solution
    σ_x = data_in.dist_params['σ_x'][0]
    σ_y = data_in.dist_params['σ_y'][0]
    μ_x = data_in.dist_params['μ_x'][0]
    μ_y = data_in.dist_params['μ_y'][0]
    z = data_in.dist_params['z'][0]
    w = data_in.dist_params['w'][0]
    simple_u = data_in.dist_params['simple_u'].numpy()[0]
    n_dist = data_in.dist_params['n_dist'].numpy()[0]
    # print('showing dist_params:', data_in.dist_params)

    # construct u_exact
    u_exact = None
    if (simple_u):  # use sigmas to construct u_exact
        def func(x, y):
            temp = 0
            for i in range(n_dist):
                temp += fd.exp(-1 * (
                    (((x-μ_x[i])**2) + ((y-μ_y[i])**2)) / w[i]
                ))
            return temp
        u_exact = func
    else:  # use w to construct u_exact
        def func(x, y):
            temp = 0
            for i in range(n_dist):
                temp += z[i] * fd.exp(-1 * (
                    (((x-μ_x[i])**2) / (σ_x[i]**2)) +
                    (((y-μ_y[i])**2) / (σ_y[i]**2))
                ))
            return temp
        u_exact = func

    # construct the helmholtz equation
    eq = None
    if problem_type == 'helmholtz':
        eq = wm.HelmholtzEqGenerator(params={
            "u_exact_func": u_exact,
        })
    elif problem_type == 'poisson':
        eq = wm.PoissonEqGenerator(params={
            "u_exact_func": u_exact,
        })

    # solution on og mesh
    og_res = eq.discretise(mesh)
    og_solver = wm.EquationSolver(params={
        "function_space": og_res["function_space"],
        "LHS": og_res["LHS"],
        "RHS": og_res["RHS"],
        "bc": og_res["bc"]
    })
    uh_og = og_solver.solve_eq()

    # solution on MA mesh
    mesh_MA.coordinates.dat.data[:] = data_in.y.detach().cpu().numpy()
    ma_res = eq.discretise(mesh_MA)
    ma_solver = wm.EquationSolver(params={
        "function_space": ma_res["function_space"],
        "LHS": ma_res["LHS"],
        "RHS": ma_res["RHS"],
        "bc": ma_res["bc"]
    })
    uh_ma = ma_solver.solve_eq()

    # solution on model mesh
    uh_model = None
    if (num_tangle == 0):
        model_res = eq.discretise(mesh_model)
        model_solver = wm.EquationSolver(params={
            "function_space": model_res["function_space"],
            "LHS": model_res["LHS"],
            "RHS": model_res["RHS"],
            "bc": model_res["bc"]
        })
        uh_model = model_solver.solve_eq()

    # a high_res mesh
    high_res_mesh = mesh_fine
    high_res_function_space = fd.FunctionSpace(
        high_res_mesh, "CG", 1)

    # exact solution on high_res mesh
    res_high_res = eq.discretise(high_res_mesh)
    u_exact = fd.interpolate(
        res_high_res["u_exact"], high_res_function_space
    )

    # projections
    uh_model_hr = None
    if (num_tangle == 0):
        uh_model_hr = fd.project(uh_model, high_res_function_space)
    uh_og_hr = fd.project(uh_og, high_res_function_space)
    uh_ma_hr = fd.project(uh_ma, high_res_function_space)

    # errornorm calculation
    error_model_mesh = -1
    if (num_tangle == 0):
        error_model_mesh = fd.errornorm(
            u_exact, uh_model_hr
        )
    error_og_mesh = fd.errornorm(
        u_exact, uh_og_hr
    )
    error_ma_mesh = fd.errornorm(
        u_exact, uh_ma_hr
    )

    return {
        "error_model_mesh": error_model_mesh,
        "error_og_mesh": error_og_mesh,
        "error_ma_mesh": error_ma_mesh,
        "u_exact": u_exact,
    }
