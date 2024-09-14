# Author: Chunyang Wang
# GitHub Username: chunyang-w

import firedrake as fd
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
import os  # noqa
import random  # noqa
import time  # noqa
import torch  # noqa
import warpmesh as wm  # noqa

import pandas as pd  # noqa

from pprint import pprint  # noqa
from torch_geometric.loader import DataLoader
from warpmesh.model.train_util import generate_samples, construct_graph


def get_log_og(log_path, idx):
    """
    Read log file from dataset log dir and return value in it
    """
    df = pd.read_csv(os.path.join(log_path, f"log{idx}.csv"))
    return {
        "error_og": df["error_og"][0],
        "error_adapt": df["error_adapt"][0],
        "time": df["time"][0],
    }


def get_first_entry(dataset, target_idx):
    for i in range(len(dataset)):
        raw_data_path = dataset.file_names[i]
        raw_data = np.load(raw_data_path, allow_pickle=True).item()
        # pprint(raw_data)
        print(raw_data.get("idx"), " ", target_idx)
        # print(raw_data.get('t'))
        if raw_data.get("idx") == target_idx:
            return i


class BurgersEvaluator:
    """
    Solves the Burgers equation
    Input:
    - mesh: The mesh on which to solve the equation.
    - dist_params: The parameters of the Gaussian distribution.

    """

    def __init__(
        self,
        mesh,
        mesh_fine,
        mesh_new,
        dataset,
        model,
        eval_dir,
        ds_root,
        idx,
        **kwargs,
    ):  # noqa
        """
        Initialise the solver.
        kwargs:
        - nu: The viscosity of the fluid.
        - dt: The time interval.
        """
        self.device = kwargs.pop("device", "cuda")
        self.model_used = kwargs.pop("model_used", "MRTransformer")
        # Mesh
        self.mesh = mesh
        self.mesh_fine = mesh_fine
        self.mesh_new = mesh_new
        # evaluation vars
        self.dataset = dataset  # dataset containing all data
        self.model = model  # the NN model
        self.eval_dir = eval_dir  # evaluation root dir
        self.ds_root = ds_root
        self.log_path = os.path.join(eval_dir, "log")
        self.plot_path = os.path.join(eval_dir, "plot")
        self.plot_more_path = os.path.join(eval_dir, "plot_more")
        self.plot_data_path = os.path.join(eval_dir, "plot_data")
        self.idx = idx
        # coordinates
        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = (
            self.mesh_fine.coordinates.vector().array().reshape(-1, 2)
        )  # noqa
        self.best_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.adapt_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        self.error_adapt_list = []
        self.error_og_list = []
        self.best_error_iter = 0

        # X and Y coordinates
        self.x, self.y = fd.SpatialCoordinate(mesh)
        self.x_fine, self.y_fine = fd.SpatialCoordinate(self.mesh_fine)
        # Function spaces
        self.P1 = fd.FunctionSpace(mesh, "CG", 1)
        self.P2 = fd.FunctionSpace(mesh, "CG", 2)
        self.P1_vec = fd.VectorFunctionSpace(mesh, "CG", 1)
        self.P2_vec = fd.VectorFunctionSpace(mesh, "CG", 2)
        self.P1_ten = fd.TensorFunctionSpace(mesh, "CG", 1)
        self.P2_ten = fd.TensorFunctionSpace(mesh, "CG", 2)

        self.P1_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        self.P2_vec_fine = fd.VectorFunctionSpace(self.mesh_fine, "CG", 2)
        self.phi_p2_vec_fine = fd.TestFunction(self.P2_vec_fine)

        # Test functions
        self.phi = fd.TestFunction(self.P1)
        self.phi_p2_vec = fd.TestFunction(self.P2_vec)

        self.trial_fine = fd.TrialFunction(self.P1_fine)
        self.phi_fine = fd.TestFunction(self.P1_fine)

        # buffer
        self.u_fine_buffer = fd.Function(self.P2_vec_fine)
        self.coarse_adapt = fd.Function(self.P1_vec)
        self.coarse_2_fine = fd.Function(self.P2_vec_fine)
        self.coarse_2_fine_original = fd.Function(self.P2_vec_fine)

        # simulation params
        self.nu = kwargs.pop("nu", 1e-3)
        self.gauss_list = kwargs.pop("gauss_list", None)
        self.dt = kwargs.get("dt", 1.0 / 30)
        self.sim_len = kwargs.get("T", 2.0)
        self.T = self.sim_len
        self.dtc = fd.Constant(self.dt)

        self.u_init = 0
        self.u_init_fine = 0
        num_of_gauss = len(self.gauss_list)
        for counter in range(num_of_gauss):
            c_x, c_y, w = (
                self.gauss_list[counter]["cx"],
                self.gauss_list[counter]["cy"],
                self.gauss_list[counter]["w"],
            )  # noqa
            self.u_init += fd.exp(-((self.x - c_x) ** 2 + (self.y - c_y) ** 2) / w)  # noqa
            self.u_init_fine += fd.exp(
                -((self.x_fine - c_x) ** 2 + (self.y_fine - c_y) ** 2) / w
            )  # noqa

        # solution vars
        self.u_og = fd.Function(self.P2_vec)  # u_{0}
        self.u = fd.Function(self.P2_vec)  # u_{n+1}
        self.u_ = fd.Function(self.P2_vec)  # u_{n}
        self.F = (
            fd.inner((self.u - self.u_) / self.dtc, self.phi_p2_vec)
            + fd.inner(fd.dot(self.u, fd.nabla_grad(self.u)), self.phi_p2_vec)
            + self.nu * fd.inner(fd.grad(self.u), fd.grad(self.phi_p2_vec))
        ) * fd.dx(domain=self.mesh)

        self.u_fine = fd.Function(self.P2_vec_fine)  # u_{0}
        self.u_fine_ = fd.Function(self.P2_vec_fine)  # u_{n+1}
        self.F_fine = (
            fd.inner((self.u_fine - self.u_fine_) / self.dtc, self.phi_p2_vec_fine)
            + fd.inner(
                fd.dot(self.u_fine, fd.nabla_grad(self.u_fine)),
                self.phi_p2_vec_fine,  # noqa
            )
            + self.nu * fd.inner(fd.grad(self.u_fine), fd.grad(self.phi_p2_vec_fine))
        ) * fd.dx(domain=self.mesh_fine)

        # initial vals
        self.initial_velocity = fd.as_vector([self.u_init, 0])
        self.initial_velocity_fine = fd.as_vector([self.u_init_fine, 0])

        self.u.project(self.initial_velocity)
        self.u_.assign(self.u)
        self.u_og.assign(self.u)

        ic_fine = fd.project(self.initial_velocity_fine, self.P2_vec_fine)
        self.u_fine.assign(ic_fine)
        self.u_fine_.assign(ic_fine)
        self.u_fine_buffer.assign(ic_fine)

        # solver params
        self.sp = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    def project_u_(self):
        self.u_.project(self.u_fine_buffer)

    def eval_problem(self):
        """
        Solves the Burgers equation.
        """
        print("target index", self.idx)
        idx_start = get_first_entry(self.dataset, self.idx)
        print("idx_start: ", idx_start)
        i = 0
        t = 0.0
        self.step = 0
        self.best_error_iter = 0
        res = {
            "deform_loss": None,  # 1. nodal position loss
            "tangled_element": None,  # 2. tangled elements on a mesh  # noqa
            "error_og": None,  # 3. PDE error on original uniform mesh  # noqa
            "error_model": None,  # 4. PDE error on model generated mesh   # noqa
            "error_ma": None,  # 5. PDE error on MA generated mesh      # noqa
            "error_reduction_MA": None,  # 6. PDE error reduced by using MA mesh  # noqa
            "error_reduction_model": None,  # 7. PDE error reduced by using model mesh  # noqa
            "time_consumption_model": None,  # 8. time consumed generating mesh inferenced by the model  # noqa
            "time_consumption_MA": None,  # 9. time consumed generating mesh by Monge-Ampere method  # noqa
            "acceration_ratio": None,  # 10. time_consumption_ma / time_consumption_model  # noqa
        }
        while t < self.T - 0.5 * self.dt:
            # get model raw file:
            cur_step = idx_start + i
            raw_data_path = self.dataset.file_names[cur_step]
            raw_data = np.load(raw_data_path, allow_pickle=True).item()
            # get sample for item
            self.error_adapt_list = []
            self.error_og_list = []
            print("step: {}, t: {}".format(self.step, t))
            # solve on fine mesh
            fd.solve(self.F_fine == 0, self.u_fine)
            # PDE error measuring
            print("cur_step: ", cur_step)
            print("compare:", t, raw_data.get("t"), raw_data.get("idx"), self.idx)
            if (abs(t - raw_data.get("t")) < 1e-5) and raw_data.get("idx") == self.idx:
                print("in here", t, raw_data.get("t"), raw_data.get("idx"))
                sample = next(
                    iter(
                        DataLoader(
                            [self.dataset[cur_step]], batch_size=1, shuffle=False
                        )
                    )
                )
                self.model.eval()
                bs = 1
                sample = sample.to(self.device)
                self.model = self.model.to(self.device)
                with torch.no_grad():
                    start = time.perf_counter()
                    if self.model_used == "MRTransformer" or self.model_used == "MRT":
                        # Create mesh query for deformer, seperate from the original mesh as feature for encoder
                        mesh_query_x = (
                            sample.mesh_feat[:, 0].view(-1, 1).detach().clone()
                        )
                        mesh_query_y = (
                            sample.mesh_feat[:, 1].view(-1, 1).detach().clone()
                        )
                        mesh_query_x.requires_grad = True
                        mesh_query_y.requires_grad = True
                        mesh_query = torch.cat([mesh_query_x, mesh_query_y], dim=-1)

                        num_nodes = mesh_query.shape[-2] // bs
                        # Generate random mesh queries for unsupervised learning
                        sampled_queries = generate_samples(
                            bs=bs,
                            num_samples_per_mesh=num_nodes,
                            num_meshes=5,
                            data=sample,
                            device=self.device,
                        )
                        sampled_queries_edge_index = construct_graph(
                            sampled_queries[:, :, :2], num_neighbors=6
                        )

                        mesh_sampled_queries_x = (
                            sampled_queries[:, :, 0].view(-1, 1).detach()
                        )
                        mesh_sampled_queries_y = (
                            sampled_queries[:, :, 1].view(-1, 1).detach()
                        )
                        mesh_sampled_queries_x.requires_grad = True
                        mesh_sampled_queries_y.requires_grad = True
                        mesh_sampled_queries = torch.cat(
                            [mesh_sampled_queries_x, mesh_sampled_queries_y], dim=-1
                        ).view(-1, 2)

                        coord_ori_x = sample.mesh_feat[:, 0].view(-1, 1)
                        coord_ori_y = sample.mesh_feat[:, 1].view(-1, 1)
                        coord_ori_x.requires_grad = True
                        coord_ori_y.requires_grad = True
                        coord_ori = torch.cat([coord_ori_x, coord_ori_y], dim=-1)

                        num_nodes = coord_ori.shape[-2] // bs
                        input_q = sample.mesh_feat[:, :4]
                        input_kv = generate_samples(
                            bs=bs,
                            num_samples_per_mesh=num_nodes,
                            data=sample,
                            device=self.device,
                        )
                        # print(f"batch size: {bs}, num_nodes: {num_nodes}, input q", input_q.shape, "input_kv ", input_kv.shape)

                        (output_coord_all, output, out_monitor), (phix, phiy) = (
                            self.model(
                                sample,
                                input_q,
                                input_q,
                                mesh_query,
                                sampled_queries=None,
                                sampled_queries_edge_index=None,
                            )
                        )
                        # (output_coord_all, output, out_monitor), (phix, phiy) = model(data, input_q, input_kv, mesh_query, sampled_queries, sampled_queries_edge_index)
                        out = output_coord_all[: num_nodes * bs]
                    elif self.model_used == "M2N":
                        out = self.model(sample)
                    elif self.model_used == "MRN":
                        out = self.model(sample)
                    else:
                        raise Exception(f"model {self.model_used} not implemented.")
                    end = time.perf_counter()
                    dur_ms = (end - start) * 1000

                # check mesh integrity - Only perform evaluation on non-tangling mesh  # noqa
                num_tangle = wm.get_sample_tangle(out, sample.x[:, :2], sample.face)  # noqa
                if isinstance(num_tangle, torch.Tensor):
                    num_tangle = num_tangle.item()
                if num_tangle > 0:  # has tangled elems:
                    res["tangled_element"] = num_tangle
                    res["error_model"] = -1
                else:  # mesh is valid, perform evaluation: 1.
                    res["tangled_element"] = num_tangle
                    # perform PDE error analysis on model generated mesh
                    self.adapt_coord = out.detach().cpu().numpy()
                    _, error_model = self.get_error()
                    res["error_model"] = error_model

                # get time_MA by reading log file
                res["time_consumption_MA"] = get_log_og(
                    os.path.join(self.ds_root, "log"), (cur_step + 1)
                )["time"]

                # metric calculation
                res["deform_loss"] = 1000 * torch.nn.L1Loss()(out, sample.y).item()
                res["time_consumption_model"] = dur_ms

                res["acceration_ratio"] = (
                    res["time_consumption_MA"] / res["time_consumption_model"]
                )  # noqa

                # solution calculation
                mesh_new = self.mesh_new
                self.adapt_coord = sample.y.cpu()
                mesh_new.coordinates.dat.data[:] = self.adapt_coord
                # calculate solution on original mesh
                self.mesh.coordinates.dat.data[:] = self.init_coord
                self.project_u_()
                fd.solve(self.F == 0, self.u)
                function_space = fd.FunctionSpace(self.mesh, "CG", 1)
                uh_0 = fd.Function(function_space)
                uh_0.project(self.u[0])

                # calculate solution on adapted mesh
                self.mesh.coordinates.dat.data[:] = self.adapt_coord
                self.project_u_()
                fd.solve(self.F == 0, self.u)
                function_space_new = fd.FunctionSpace(mesh_new, "CG", 1)
                function_space_vec_new = fd.VectorFunctionSpace(mesh_new, "CG", 1)
                uh_new = fd.Function(function_space_vec_new)
                uh_new.project(self.u)
                uh_new_0 = fd.Function(function_space_new)
                uh_new_0.project(uh_new[0])

                error_og, error_adapt = self.get_error()
                print("error_og: {}, error_adapt: {}".format(error_og, error_adapt))  # noqa

                res["error_og"] = error_og
                res["error_ma"] = error_adapt
                res["error_reduction_MA"] = (res["error_og"] - res["error_ma"]) / res[
                    "error_og"
                ]  # noqa
                res["error_reduction_model"] = (
                    res["error_og"] - res["error_model"]
                ) / res["error_og"]  # noqa

                # save file
                df = pd.DataFrame(res, index=[0])
                df.to_csv(os.path.join(self.log_path, f"log{self.idx}_{cur_step}.csv"))  # noqa

                # plot compare mesh
                compare_plot = wm.plot_mesh_compare_benchmark(
                    out.detach().cpu().numpy(),
                    sample.y.detach().cpu().numpy(),
                    sample.face.detach().cpu().numpy(),
                    res["deform_loss"],
                    res["error_model"],
                    res["error_reduction_model"],
                    res["error_ma"],
                    res["error_reduction_MA"],
                    res["tangled_element"],
                )
                compare_plot.savefig(
                    os.path.join(self.plot_path, f"plot_{self.idx}_{cur_step}.png")
                )  # noqa
                # put coords back to original position (for u sampling)
                self.mesh.coordinates.dat.data[:] = self.init_coord

                # 3D plot of model solution
                # more detailed plot - 3d plot and 2d plot with mesh
                fig = plt.figure(figsize=(8, 8))

                # 3D plot of MA solution                    TODO
                ax1 = fig.add_subplot(2, 2, 1, projection="3d")
                ax1.set_title("MA Solution (3D)")
                fd.trisurf(uh_new_0, axes=ax1)
                if num_tangle == 0:
                    # solve on coarse adapt mesh
                    function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)  # noqa
                    self.mesh.coordinates.dat.data[:] = out.detach().cpu().numpy()  # noqa
                    function_space = fd.FunctionSpace(self.mesh, "CG", 1)
                    self.project_u_()
                    fd.solve(self.F == 0, self.u)
                    u_adapt_coarse_0 = fd.Function(function_space)
                    u_adapt_coarse_0.project(self.u[0])
                    # old
                    # self.adapt_coord = out.detach().cpu().numpy()
                    # self.mesh.coordinates.dat.data[:] = self.adapt_coord
                    # self.mesh_new.coordinates.dat.data[:] = self.adapt_coord
                    # self.project_u_()
                    # self.solve_u(self.t)
                    # function_space_new = fd.FunctionSpace(self.mesh_new, "CG", 1)  # noqa
                    # uh_model = fd.Function(function_space_new).project(self.u_cur)  # noqa
                    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
                    ax2.set_title("Model Solution (3D)")
                    fd.trisurf(u_adapt_coarse_0, axes=ax2)

                    # 2d plot and mesh for Model            TODO
                    ax4 = fig.add_subplot(2, 2, 4)
                    ax4.set_title("Soultion on Model mesh")
                    fd.tripcolor(u_adapt_coarse_0, cmap="coolwarm", axes=ax4)
                    self.mesh_new.coordinates.dat.data[:] = out.detach().cpu().numpy()  # noqa
                    fd.triplot(self.mesh_new, axes=ax4)

                # 2d plot and mesh for MA
                ax3 = fig.add_subplot(2, 2, 3)
                ax3.set_title("Soultion on MA mesh")
                fd.tripcolor(uh_new_0, cmap="coolwarm", axes=ax3)
                self.mesh_new.coordinates.dat.data[:] = sample.y.detach().cpu().numpy()
                fd.triplot(self.mesh_new, axes=ax3)

                fig.savefig(
                    os.path.join(self.plot_more_path, f"plot_{self.idx}_{cur_step}.png")
                )  # noqa
                plt.close()
                i += 1
            # step forward in time
            self.u_fine_.assign(self.u_fine)
            # self.u_fine_buffer.project(self.u)
            self.u_fine_buffer.assign(self.u_fine)
            # fd.triplot(self.u_fine)
            # plt.show()
            # self.u_.assign(self.u)
            t += self.dt
            self.step += 1

        return

    def get_error(self):
        # print("get_error: u_ sum is: ", np.sum(self.u_.dat.data[:]))
        function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        # solve on fine mesh
        fd.solve(self.F_fine == 0, self.u_fine)
        u_fine_0 = fd.Function(function_space_fine)
        u_f = u_fine_0.project(self.u_fine[0])
        # print('u_f sum: ', np.sum(u_f.dat.data[:]))

        # solve on coarse mesh
        self.mesh.coordinates.dat.data[:] = self.init_coord
        function_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.project_u_()
        # print("og u_ sum: ", np.sum(self.u_.dat.data[:]))
        fd.solve(self.F == 0, self.u)
        u_0_fine = fd.Function(function_space_fine)
        u_0_coarse = fd.Function(function_space)
        u_0_coarse.project(self.u[0])
        u_0_fine.project(u_0_coarse)
        # print('u_0_fine sum 1: ', np.sum(u_0_fine.dat.data[:]))
        error_og = fd.errornorm(u_0_fine, u_f, norm_type="L2")

        # solve on coarse adapt mesh
        self.mesh.coordinates.dat.data[:] = self.adapt_coord
        function_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.project_u_()
        # print("adapt u_ sum: ", np.sum(self.u_.dat.data[:]))
        fd.solve(self.F == 0, self.u)
        u_adapt_fine_0 = fd.Function(function_space_fine)
        u_adapt_coarse_0 = fd.Function(function_space)
        u_adapt_coarse_0.project(self.u[0])
        u_adapt_fine_0.project(u_adapt_coarse_0)
        # print('u sum 2: ', np.sum(u_adapt_fine_0.dat.data[:]))
        error_adapt = fd.errornorm(u_adapt_fine_0, u_f, norm_type="L2")

        self.mesh.coordinates.dat.data[:] = self.init_coord

        return error_og, error_adapt

    def make_log_dir(self):
        wm.mkdir_if_not_exist(self.log_path)

    def make_plot_dir(self):
        wm.mkdir_if_not_exist(self.plot_path)

    def make_plot_more_dir(self):
        wm.mkdir_if_not_exist(self.plot_more_path)

    def make_plot_data_dir(self):
        wm.mkdir_if_not_exist(self.plot_data_path)
