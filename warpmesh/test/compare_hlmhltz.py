# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import firedrake as fd
import warpmesh as wm
import matplotlib.pyplot as plt  # noqa

__all__ = ["compare_error"]


def compare_error(
    data_in,
    mesh,
    mesh_fine,
    mesh_model,
    mesh_MA,
    num_tangle,
    model_name,
    problem_type="helmholtz",
):
    # read in params used to construct the analytical solution
    σ_x = data_in.dist_params["σ_x"][0]
    σ_y = data_in.dist_params["σ_y"][0]
    μ_x = data_in.dist_params["μ_x"][0]
    μ_y = data_in.dist_params["μ_y"][0]
    z = data_in.dist_params["z"][0]
    w = data_in.dist_params["w"][0]
    simple_u = data_in.dist_params["simple_u"].cpu().numpy()[0]
    n_dist = data_in.dist_params["n_dist"].cpu().numpy()[0]
    # print('showing dist_params:', data_in.dist_params)
    # print("data in ", data_in)

    if model_name == "MRTransformer":
        model_name = "M2T"

    # construct u_exact
    u_exact = None
    if simple_u:  # use sigmas to construct u_exact

        def func(x, y):
            temp = 0
            for i in range(n_dist):
                temp += fd.exp(
                    -1 * ((((x - μ_x[i]) ** 2) + ((y - μ_y[i]) ** 2)) / w[i])
                )
            return temp

        u_exact = func
    else:  # use w to construct u_exact

        def func(x, y):
            temp = 0
            for i in range(n_dist):
                temp += z[i] * fd.exp(
                    -1
                    * (
                        (((x - μ_x[i]) ** 2) / (σ_x[i] ** 2))
                        + (((y - μ_y[i]) ** 2) / (σ_y[i] ** 2))
                    )
                )
            return temp

        u_exact = func

    # construct the helmholtz equation
    eq = None
    if problem_type == "helmholtz":
        eq = wm.HelmholtzEqGenerator(
            params={
                "u_exact_func": u_exact,
            }
        )
    elif problem_type == "poisson":
        eq = wm.PoissonEqGenerator(
            params={
                "u_exact_func": u_exact,
            }
        )

    # solution on og mesh
    og_res = eq.discretise(mesh)
    og_solver = wm.EquationSolver(
        params={
            "function_space": og_res["function_space"],
            "LHS": og_res["LHS"],
            "RHS": og_res["RHS"],
            "bc": og_res["bc"],
        }
    )
    uh_og = og_solver.solve_eq()

    # solution on MA mesh
    mesh_MA.coordinates.dat.data[:] = data_in.y.detach().cpu().numpy()
    ma_res = eq.discretise(mesh_MA)
    ma_solver = wm.EquationSolver(
        params={
            "function_space": ma_res["function_space"],
            "LHS": ma_res["LHS"],
            "RHS": ma_res["RHS"],
            "bc": ma_res["bc"],
        }
    )
    uh_ma = ma_solver.solve_eq()

    # solution on model mesh
    uh_model = None
    if num_tangle == 0:
        model_res = eq.discretise(mesh_model)
        model_solver = wm.EquationSolver(
            params={
                "function_space": model_res["function_space"],
                "LHS": model_res["LHS"],
                "RHS": model_res["RHS"],
                "bc": model_res["bc"],
            }
        )
        uh_model = model_solver.solve_eq()
    
    # a high_res mesh
    high_res_mesh = mesh_fine
    high_res_function_space = fd.FunctionSpace(high_res_mesh, "CG", 1)

    # exact solution on high_res mesh
    res_high_res = eq.discretise(high_res_mesh)
    uh_exact = fd.interpolate(res_high_res["u_exact"], high_res_function_space)

    fig, plot_data_dict = wm.plot_compare(
        mesh_fine,
        mesh,
        mesh_MA,
        mesh_model,
        uh_exact,
        uh_og,
        uh_ma,
        uh_model,
        data_in.monitor_val[:, 0].detach().cpu().numpy(),
        data_in.monitor_val[:, 0].detach().cpu().numpy(),
        num_tangle,
        model_name,
    )

    # # a high_res mesh
    # high_res_mesh = mesh_fine
    # high_res_function_space = fd.FunctionSpace(high_res_mesh, "CG", 1)

    # # exact solution on high_res mesh
    # res_high_res = eq.discretise(high_res_mesh)
    # u_exact = fd.interpolate(res_high_res["u_exact"], high_res_function_space)

    # # projections
    # uh_model_hr = None
    # if num_tangle == 0:
    #     uh_model_hr = fd.project(uh_model, high_res_function_space)
    # uh_og_hr = fd.project(uh_og, high_res_function_space)
    # uh_ma_hr = fd.project(uh_ma, high_res_function_space)

    # # errornorm calculation
    # error_model_mesh = -1
    # if num_tangle == 0:
    #     error_model_mesh = fd.errornorm(u_exact, uh_model_hr)
    # error_og_mesh = fd.errornorm(u_exact, uh_og_hr)
    # error_ma_mesh = fd.errornorm(u_exact, uh_ma_hr)

    # # ====  Plot mesh, solution, error ======================
    # plot_data_dict = {}

    # rows, cols = 3, 4
    # fig, ax = plt.subplots(
    #     rows, cols, figsize=(cols * 5, rows * 5), layout="compressed"
    # )

    # # High resolution mesh
    # fd.triplot(mesh_fine, axes=ax[0, 0])
    # ax[0, 0].set_title(f"High resolution Mesh (100 x 100)")
    # # Orginal low resolution uniform mesh
    # fd.triplot(mesh, axes=ax[0, 1])
    # ax[0, 1].set_title(f"Original uniform Mesh")
    # # Adapted mesh (MA)
    # fd.triplot(mesh_MA, axes=ax[0, 2])
    # ax[0, 2].set_title(f"Adapted Mesh (MA)")
    # # Adapted mesh (Model)
    # fd.triplot(mesh_model, axes=ax[0, 3])
    # ax[0, 3].set_title(f"Adapted Mesh ({model_name})")

    # plot_data_dict["mesh_ma"] = mesh_MA.coordinates.dat.data[:]
    # plot_data_dict["mesh_model"] = mesh_model.coordinates.dat.data[:]

    # cmap = "seismic"

    # u_exact_max = u_exact.dat.data[:].max()
    # u_og_max = uh_og.dat.data[:].max()
    # u_ma_max = uh_ma.dat.data[:].max()
    # u_model_max = uh_model.dat.data[:].max() if uh_model else float("-inf")
    # solution_v_max = max(u_exact_max, u_og_max, u_ma_max, u_model_max)

    # u_exact_min = u_exact.dat.data[:].min()
    # u_og_min = uh_og.dat.data[:].min()
    # u_ma_min = uh_ma.dat.data[:].min()
    # u_model_min = uh_model.dat.data[:].min() if uh_model else float("inf")
    # solution_v_min = min(u_exact_min, u_og_min, u_ma_min, u_model_min)

    # # Solution on high resolution mesh
    # cb = fd.tripcolor(
    #     u_exact, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 0]
    # )
    # ax[1, 0].set_title(f"Solution on High Resolution (u_exact)")
    # plt.colorbar(cb)
    # # Solution on orginal low resolution uniform mesh
    # cb = fd.tripcolor(
    #     uh_og, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 1]
    # )
    # ax[1, 1].set_title(f"Solution on uniform Mesh")
    # plt.colorbar(cb)
    # # Solution on adapted mesh (MA)
    # cb = fd.tripcolor(
    #     uh_ma, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 2]
    # )
    # ax[1, 2].set_title(f"Solution on Adapted Mesh (MA)")
    # plt.colorbar(cb)

    # if uh_model:
    #     # Solution on adapted mesh (Model)
    #     cb = fd.tripcolor(
    #         uh_model, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 3]
    #     )
    #     ax[1, 3].set_title(f"Solution on Adapted Mesh ({model_name})")
    #     plt.colorbar(cb)
    #     plot_data_dict["u_model"] = uh_model.dat.data[:]

    # plot_data_dict["u_exact"] = u_exact.dat.data[:]
    # plot_data_dict["u_original"] = uh_og.dat.data[:]
    # plot_data_dict["u_ma"] = uh_ma.dat.data[:]
    # plot_data_dict["u_v_max"] = solution_v_max
    # plot_data_dict["u_v_min"] = solution_v_min

    # err_orignal_mesh = fd.assemble(uh_og_hr - u_exact)
    # err_adapted_mesh_ma = fd.assemble(uh_ma_hr - u_exact)

    # if uh_model_hr:
    #     err_adapted_mesh_model = fd.assemble(uh_model_hr - u_exact)
    #     err_abs_max_val_adapted_mesh_model = max(
    #         abs(err_adapted_mesh_model.dat.data[:].max()),
    #         abs(err_adapted_mesh_model.dat.data[:].min()),
    #     )
    # else:
    #     err_abs_max_val_adapted_mesh_model = 0.0

    # err_abs_max_val_ori = max(
    #     abs(err_orignal_mesh.dat.data[:].max()), abs(err_orignal_mesh.dat.data[:].min())
    # )
    # err_abs_max_val_adapted_ma = max(
    #     abs(err_adapted_mesh_ma.dat.data[:].max()),
    #     abs(err_adapted_mesh_ma.dat.data[:].min()),
    # )

    # err_abs_max_val = max(
    #     max(err_abs_max_val_ori, err_abs_max_val_adapted_ma),
    #     err_abs_max_val_adapted_mesh_model,
    # )
    # err_v_max = err_abs_max_val
    # err_v_min = -err_v_max

    # # Visualize the monitor values of MA
    # monitor_val = data_in.monitor_val
    # monitor_val_vis_holder = fd.Function(ma_res["function_space"])
    # monitor_val_vis_holder.dat.data[:] = monitor_val[:, 0].detach().cpu().numpy()
    # # Monitor values
    # cb = fd.tripcolor(monitor_val_vis_holder, cmap=cmap, axes=ax[2, 0])
    # ax[2, 0].set_title(f"Monitor values")
    # plt.colorbar(cb)
    # # Error on orginal low resolution uniform mesh
    # cb = fd.tripcolor(
    #     err_orignal_mesh, cmap=cmap, axes=ax[2, 1], vmax=err_v_max, vmin=err_v_min
    # )
    # ax[2, 1].set_title(f"Error (u-u_exact) uniform Mesh | L2 Norm: {error_og_mesh:.5f}")
    # plt.colorbar(cb)
    # # Error on adapted mesh (MA)
    # cb = fd.tripcolor(
    #     err_adapted_mesh_ma, cmap=cmap, axes=ax[2, 2], vmax=err_v_max, vmin=err_v_min
    # )
    # ax[2, 2].set_title(
    #     f"Error (u-u_exact) MA| L2 Norm: {error_ma_mesh:.5f} | {(error_og_mesh-error_ma_mesh)/error_og_mesh*100:.2f}%"
    # )
    # plt.colorbar(cb)

    # if uh_model_hr:
    #     # Error on adapted mesh (Model)
    #     cb = fd.tripcolor(
    #         err_adapted_mesh_model,
    #         cmap=cmap,
    #         axes=ax[2, 3],
    #         vmax=err_v_max,
    #         vmin=err_v_min,
    #     )
    #     ax[2, 3].set_title(
    #         f"Error (u-u_exact) {model_name}| L2 Norm: {error_model_mesh:.5f} | {(error_og_mesh-error_model_mesh)/error_og_mesh*100:.2f}%"
    #     )
    #     plt.colorbar(cb)

    #     plot_data_dict["error_map_model"] = err_adapted_mesh_model.dat.data[:]
    #     plot_data_dict["error_norm_model"] = error_model_mesh

    # plot_data_dict["monitor_values"] = monitor_val_vis_holder.dat.data[:]
    # plot_data_dict["error_map_original"] = err_orignal_mesh.dat.data[:]
    # plot_data_dict["error_map_ma"] = err_adapted_mesh_ma.dat.data[:]

    # plot_data_dict["error_norm_original"] = error_og_mesh
    # plot_data_dict["error_norm_ma"] = error_ma_mesh

    # # For visualization
    # plot_data_dict["error_v_max"] = err_v_max

    # for rr in range(rows):
    #     for cc in range(cols):
    #         ax[rr, cc].set_aspect("equal", "box")

    error_og_mesh = plot_data_dict["error_norm_original"]
    error_ma_mesh = plot_data_dict["error_norm_ma"]
    error_model_mesh = plot_data_dict["error_norm_model"]

    return {
        "error_model_mesh": error_model_mesh,
        "error_og_mesh": error_og_mesh,
        "error_ma_mesh": error_ma_mesh,
        "u_exact": u_exact,
        "plot_more": fig,
        "plot_data_dict": plot_data_dict,
    }
