# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import firedrake as fd
import warpmesh as wm
import matplotlib.pyplot as plt # noqa

__all__ = ['compare_error']


def compare_error(data_in,
                  mesh, mesh_fine, mesh_model, mesh_MA, num_tangle, model_name,
                  problem_type='helmholtz'
                  ):
    # read in params used to construct the analytical solution
    σ_x = data_in.dist_params['σ_x'][0]
    σ_y = data_in.dist_params['σ_y'][0]
    μ_x = data_in.dist_params['μ_x'][0]
    μ_y = data_in.dist_params['μ_y'][0]
    z = data_in.dist_params['z'][0]
    w = data_in.dist_params['w'][0]
    simple_u = data_in.dist_params['simple_u'].cpu().numpy()[0]
    n_dist = data_in.dist_params['n_dist'].cpu().numpy()[0]
    # print('showing dist_params:', data_in.dist_params)

    if model_name == 'MRTransformer':
        model_name = 'M2T'

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

    # fig = plt.figure(figsize=(8, 8))
    # # 3D plot of MA solution
    # ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # ax1.set_title('MA Solution (3D)')
    # fd.trisurf(uh_ma, axes=ax1)
    # # 3D plot of model solution
    # if (num_tangle == 0):
    #     ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    #     ax2.set_title("Model Solution (3D)")
    #     fd.trisurf(uh_model, axes=ax2)

    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.set_title("Solution on MA Mesh")
    # fd.tripcolor(uh_ma, cmap='coolwarm', axes=ax3)
    # fd.triplot(mesh_MA, axes=ax3)

    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.set_title("Solution on Model Mesh")
    # if (num_tangle == 0):
    #     fd.tripcolor(uh_model, cmap='coolwarm', axes=ax4)
    # fd.triplot(mesh_model, axes=ax4)

    # ====  Plot mesh, solution, error ======================
    rows, cols = 3, 4
    fig, ax = plt.subplots(rows, cols, figsize=(cols*5, rows*5 ), layout='compressed')

    # High resolution mesh
    fd.triplot(mesh_fine, axes=ax[0, 0])
    ax[0, 0].set_title(f"High resolution Mesh (100 x 100)")
    # Orginal low resolution uniform mesh
    fd.triplot(mesh, axes=ax[0, 1])
    ax[0, 1].set_title(f"Original uniform Mesh")
    # Adapted mesh (MA)
    fd.triplot(mesh_MA, axes=ax[0, 2])
    ax[0, 2].set_title(f"Adapted Mesh (MA)")
    # Adapted mesh (Model)
    fd.triplot(mesh_model, axes=ax[0, 3])
    ax[0, 3].set_title(f"Adapted Mesh ({model_name})")

    cmap = 'seismic'
    # Solution on high resolution mesh
    cb = fd.tripcolor(u_exact, cmap=cmap, axes=ax[1, 0])
    ax[1, 0].set_title(f"Solution on High Resolution (u_exact)")
    plt.colorbar(cb)
    # Solution on orginal low resolution uniform mesh
    cb = fd.tripcolor(uh_og, cmap=cmap, axes=ax[1, 1])
    ax[1, 1].set_title(f"Solution on uniform Mesh")
    plt.colorbar(cb)
    # Solution on adapted mesh (MA)
    cb = fd.tripcolor(uh_ma, cmap=cmap, axes=ax[1, 2])
    ax[1, 2].set_title(f"Solution on Adapted Mesh (MA)")
    plt.colorbar(cb)

    if uh_model:
        # Solution on adapted mesh (Model)
        cb = fd.tripcolor(uh_model, cmap=cmap, axes=ax[1, 3])
        ax[1, 3].set_title(f"Solution on Adapted Mesh ({model_name})")
        plt.colorbar(cb)


    err_orignal_mesh = fd.assemble(uh_og_hr - u_exact)
    err_adapted_mesh_ma = fd.assemble(uh_ma_hr - u_exact)

    if uh_model_hr:
        err_adapted_mesh_model = fd.assemble(uh_model_hr - u_exact)
        err_abs_max_val_adapted_mesh_model = max(abs(err_adapted_mesh_model.dat.data[:].max()), abs(err_adapted_mesh_model.dat.data[:].min()))
    else:
        err_abs_max_val_adapted_mesh_model = 0.0

    err_abs_max_val_ori = max(abs(err_orignal_mesh.dat.data[:].max()), abs(err_orignal_mesh.dat.data[:].min()))
    err_abs_max_val_adapted_ma = max(abs(err_adapted_mesh_ma.dat.data[:].max()), abs(err_adapted_mesh_ma.dat.data[:].min()))
    

    err_abs_max_val = max(max(err_abs_max_val_ori, err_abs_max_val_adapted_ma), err_abs_max_val_adapted_mesh_model)
    err_v_max = err_abs_max_val
    err_v_min = -err_v_max
    
    # Error on high resolution mesh
    cb = fd.tripcolor(fd.assemble(u_exact - u_exact), cmap=cmap, axes=ax[2, 0], vmax=err_v_max, vmin=err_v_min)
    ax[2, 0].set_title(f"Error Map High Resolution")
    plt.colorbar(cb)
    # Error on orginal low resolution uniform mesh
    cb = fd.tripcolor(err_orignal_mesh, cmap=cmap, axes=ax[2, 1], vmax=err_v_max, vmin=err_v_min)
    ax[2, 1].set_title(f"Error (u-u_exact) uniform Mesh | L2 Norm: {error_og_mesh:.5f}")
    plt.colorbar(cb)
    # Error on adapted mesh (MA)
    cb = fd.tripcolor(err_adapted_mesh_ma, cmap=cmap, axes=ax[2, 2], vmax=err_v_max, vmin=err_v_min)
    ax[2, 2].set_title(f"Error (u-u_exact) MA| L2 Norm: {error_ma_mesh:.5f} | {(error_og_mesh-error_ma_mesh)/error_og_mesh*100:.2f}%")
    plt.colorbar(cb)

    if uh_model_hr:
        # Error on adapted mesh (Model)
        cb = fd.tripcolor(err_adapted_mesh_model, cmap=cmap, axes=ax[2, 3], vmax=err_v_max, vmin=err_v_min)
        ax[2, 3].set_title(f"Error (u-u_exact) {model_name}| L2 Norm: {error_model_mesh:.5f} | {(error_og_mesh-error_model_mesh)/error_og_mesh*100:.2f}%")
        plt.colorbar(cb)

    for rr in range(rows):
        for cc in range(cols):
            ax[rr, cc].set_aspect('equal', 'box')

    return {
        "error_model_mesh": error_model_mesh,
        "error_og_mesh": error_og_mesh,
        "error_ma_mesh": error_ma_mesh,
        "u_exact": u_exact,
        "plot_more": fig,
    }
