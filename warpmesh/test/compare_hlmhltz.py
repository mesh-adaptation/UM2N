# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import firedrake as fd
import warpmesh as wm
import movement as mv
import time
import matplotlib.pyplot as plt

__all__ = ['compare_error']


def compare_error(model, data_in, n_elem=20, plot=False):
    # read in params used to construct the analytical solution
    σ_x = data_in.dist_params['σ_x']
    σ_y = data_in.dist_params['σ_y']
    μ_x = data_in.dist_params['μ_x']
    μ_y = data_in.dist_params['μ_y']
    z = data_in.dist_params['z']
    w = data_in.dist_params['w']
    simple_u = data_in.dist_params['simple_u']
    n_dist = data_in.dist_params['n_dist']

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

    # construct f
    f_func = None
    if (simple_u):  # use sigmas to construct f
        def func(x, y):
            temp = 0
            for i in range(n_dist):
                temp += fd.exp(-1 * (
                    (((x-μ_x[i])**2) + ((y-μ_y[i])**2)) / w[i]
                ))
            return (-1 * fd.div(fd.grad(temp)) + temp)
        f_func = func
    else:  # use w to construct f
        def func(x, y):
            temp = 0
            for i in range(n_dist):
                temp += z[i] * fd.exp(-1 * (
                    (((x-μ_x[i])**2) / (σ_x[i]**2)) +
                    (((y-μ_y[i])**2) / (σ_y[i]**2))
                ))
            return (-1 * fd.div(fd.grad(temp)) + temp)
        f_func = func

    # construct the helmholtz equation
    helmholtz_eq = wm.HelmholtzGenerator(params={
        "f_func": f_func,
        "u_exact_func": u_exact,
    })

    # solve equation
    mesh = fd.UnitSquareMesh(n_elem, n_elem)
    res = helmholtz_eq.discretise(mesh)

    solver = wm.HelmholtzSolver(params={
        "function_space": res["function_space"],
        "LHS": res["LHS"],
        "RHS": res["RHS"],
        "bc": res["bc"]
    })

    mesh_gen = wm.MeshGenerator(params={
        "num_grid_x": n_elem,
        "num_grid_y": n_elem,
        "helmholtz_eq": helmholtz_eq,
        # "mesh": fd.UnitSquareMesh(n_elem, n_elem),
        "mesh": fd.RectangleMesh(
                        n_elem, n_elem, 1, 1)
    })
    # ma mesh
    start_time_ma = time.perf_counter_ns()
    ma_mesh = mesh_gen.move_mesh()
    end_time_ma = time.perf_counter_ns()
    # the model produced mesh
    start_time_model = time.perf_counter_ns()
    out = model(data_in).detach().numpy()
    end_time_model = time.perf_counter_ns()
    model_mesh = fd.UnitSquareMesh(n_elem, n_elem)
    checker = mv.MeshTanglingChecker(model_mesh, mode='warn')
    model_mesh.coordinates.dat.data[:] = out[:]
    num_tangle = checker.check()

    # solution on original mesh
    uh = solver.solve_eq()

    # solution on ma mesh
    ma_res = helmholtz_eq.discretise(ma_mesh)
    ma_solver = wm.HelmholtzSolver(params={
        "function_space": ma_res["function_space"],
        "LHS": ma_res["LHS"],
        "RHS": ma_res["RHS"],
        "bc": ma_res["bc"]
    })
    uh_ma = ma_solver.solve_eq()

    # solution on model mesh
    uh_model = None
    if (num_tangle == 0):
        model_res = helmholtz_eq.discretise(model_mesh)
        model_solver = wm.HelmholtzSolver(params={
            "function_space": model_res["function_space"],
            "LHS": model_res["LHS"],
            "RHS": model_res["RHS"],
            "bc": model_res["bc"]
        })
        uh_model = model_solver.solve_eq()

    # a high_res mesh
    high_res_mesh = fd.UnitSquareMesh(80, 80)
    high_res_function_space = fd.FunctionSpace(
        high_res_mesh, "CG", 1)

    # exact solution on high_res mesh
    res_high_res = helmholtz_eq.discretise(high_res_mesh)
    u_exact = res_high_res["u_exact"]

    # original, ma, model solutions on high_res mesh
    uh_hr = fd.project(uh, high_res_function_space)
    uh_ma_hr = fd.project(uh_ma, high_res_function_space)
    uh_model_hr = None
    if (num_tangle == 0):
        uh_model_hr = fd.project(uh_model, high_res_function_space)

    error_original_mesh = fd.errornorm(
        u_exact, uh_hr
    )
    error_ma_mesh = fd.errornorm(
        u_exact, uh_ma_hr
    )
    error_model_mesh = None
    if (num_tangle == 0):
        error_model_mesh = fd.errornorm(
            u_exact, uh_model_hr
        )

    compare_result = {
        "tangle_num": num_tangle,
        "error_original_mesh": error_original_mesh,
        "error_ma_mesh": error_ma_mesh,
        "error_model_mesh": error_model_mesh,
        "time_ma": end_time_ma - start_time_ma,
        "time_model": end_time_model - start_time_model,
        "acceleration": 100*(end_time_ma - start_time_ma) /
        (end_time_model - start_time_model),
    }

    if (plot):
        fig = plt.figure(figsize=(15, 5))
        # exact solution
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.set_title('Exact Solution', fontsize=16)
        fd.trisurf(uh, axes=ax1)
        ax1.text2D(-0.02, 0.5, f"Mesh size: {n_elem}x{n_elem}", transform=ax1.transAxes, rotation=90, verticalalignment='center', fontsize=18)  # noqa

        # error reduction on ma mesh:
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Error Reduction(MA): {:.2f}%'.format(
            100*(error_original_mesh - error_ma_mesh) / error_original_mesh  # noqa
        ), fontsize=16)
        fd.tripcolor(
            uh_ma, cmap='coolwarm', axes=ax2)
        fd.triplot(ma_mesh, axes=ax2)

        # error reduction on ma mesh:
        ax3 = fig.add_subplot(1, 3, 3)
        if (num_tangle == 0):
            ax3.set_title('Error Reduction(MRN): {:.2f}%'.format(
                100*(error_original_mesh - error_model_mesh) /
                error_original_mesh
            ), fontsize=16)
        else:
            ax3.set_title(f'Mesh tangled: ({num_tangle})', fontsize=16)
        if (num_tangle == 0):
            fd.tripcolor(
                uh_model, cmap='coolwarm', axes=ax3)
        fd.triplot(model_mesh, axes=ax3)

        return compare_result

    return compare_result
