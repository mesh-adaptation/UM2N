# Author: Chunyang Wang
# GitHub username: chunyang-w

import firedrake as fd
import os
import time
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warpmesh as wm

from pprint import pprint                               # noqa
from torch_geometric.loader import DataLoader
from warpmesh.model.train_util import generate_samples


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


class SwirlEvaluator():
    """
    Evaluate error for advection swirl problem:
        1. Solver implementation for the swirl problem
        2. Error & Time evaluation
    """
    def __init__(self, mesh, mesh_fine, mesh_new, dataset, model, eval_dir, ds_root, **kwargs):  # noqa
        """
        Init the problem:
            1. define problem on fine mesh and coarse mesh
            2. init function space on fine & coarse mesh
        """
        self.device = kwargs.pop("device", "cuda")
        # mesh vars
        self.mesh = mesh                                        # coarse mesh
        self.mesh_fine = mesh_fine                              # fine mesh
        self.mesh_new = mesh_new                                # adapted mesh
        # evaluation vars
        self.dataset = dataset              # dataset containing all data
        self.model = model                  # the NN model
        self.eval_dir = eval_dir            # evaluation root dir
        self.ds_root = ds_root
        self.log_path = os.path.join(eval_dir, "log")
        self.plot_path = os.path.join(eval_dir, "plot")
        self.plot_more_path = os.path.join(eval_dir, "plot_more")
        self.save_interval = kwargs.pop("save_interval", 5)

        # Init coords setup
        self.init_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.init_coord_fine = self.mesh_fine.coordinates.vector().array().reshape(-1, 2) # noqa
        self.best_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)
        self.adapt_coord = self.mesh.coordinates.vector().array().reshape(-1, 2)  # noqa
        # error measuring vars
        self.error_adapt_list = []
        self.error_og_list = []
        self.best_error_iter = 0

        # X and Y coordinates
        self.x, self.y = fd.SpatialCoordinate(mesh)
        self.x_fine, self.y_fine = fd.SpatialCoordinate(self.mesh_fine)

        # function space on coarse mesh
        self.scalar_space = fd.FunctionSpace(self.mesh, "CG", 1)
        self.vector_space = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        self.tensor_space = fd.TensorFunctionSpace(self.mesh, "CG", 1)
        # function space on fine mesh
        self.scalar_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        self.vector_space_fine = fd.VectorFunctionSpace(self.mesh_fine, "CG", 1)  # noqa

        # Test/Trial function on coarse mesh
        self.du_trial = fd.TrialFunction(self.scalar_space)
        self.phi = fd.TestFunction(self.scalar_space)
        # Test/Trial function on fine mesh
        self.du_trial_fine = fd.TrialFunction(self.scalar_space_fine)
        self.phi_fine = fd.TestFunction(self.scalar_space_fine)
        # normal function on coarse / fine mesh
        self.n = fd.FacetNormal(self.mesh)
        self.n_fine = fd.FacetNormal(self.mesh_fine)

        # simulation params
        self.T = kwargs.pop("T", 1)
        self.t = 0.0
        self.n_step = kwargs.pop("n_step", 500)
        self.threshold = self.T / 2                          # Time point the swirl direction get reverted  # noqa
        self.dt = self.T / self.n_step
        self.dtc = fd.Constant(self.dt)
        # initial condition params
        self.sigma = kwargs.pop("sigma", (0.05/6))
        self.alpha = kwargs.pop("alpha", 1.5)
        self.r_0 = kwargs.pop("r_0", 0.2)
        self.x_0 = kwargs.pop("x_0", 0.25)
        self.y_0 = kwargs.pop("y_0", 0.25)

        # initital condition of u on coarse / fine mesh
        u_init_exp = wm.get_u_0(self.x, self.y, self.r_0, self.x_0, self.y_0, self.sigma)  # noqa
        u_init_exp_fine = wm.get_u_0(self.x_fine, self.y_fine, self.r_0, self.x_0, self.y_0, self.sigma)  # noqa
        self.u_init = fd.Function(self.scalar_space).interpolate(u_init_exp)
        self.u_init_fine = fd.Function(self.scalar_space_fine).interpolate(u_init_exp_fine) # noqa
        # PDE vars on coarse & fine mesh
        #       solution field u
        self.u = fd.Function(self.scalar_space).assign(self.u_init)
        self.u1 = fd.Function(self.scalar_space)
        self.u2 = fd.Function(self.scalar_space)
        self.u_fine = fd.Function(self.scalar_space_fine).assign(self.u_init_fine)  # noqa
        self.u1_fine = fd.Function(self.scalar_space_fine)
        self.u2_fine = fd.Function(self.scalar_space_fine)
        self.u_in = fd.Constant(0.0)
        self.u_in_fine = fd.Constant(0.0)
        #       temp vars for saving u on coarse & fine mesh
        self.u_cur = fd.Function(self.scalar_space)             # solution from current time step  # noqa
        self.u_cur_fine = fd.Function(self.scalar_space_fine)
        self.u_hess = fd.Function(self.scalar_space)            # buffer for hessian solver usage  # noqa
        #       buffers
        self.u_fine_buffer = fd.Function(self.scalar_space_fine).assign(self.u_init_fine)  # noqa
        self.coarse_adapt = fd.Function(self.scalar_space)
        self.coarse_2_fine = fd.Function(self.scalar_space_fine)
        self.coarse_2_fine_original = fd.Function(self.scalar_space_fine)

        #       velocity field - the swirl: c
        self.c = fd.Function(self.vector_space)
        self.c_fine = fd.Function(self.vector_space_fine)
        self.cn = 0.5*(fd.dot(self.c, self.n) + abs(fd.dot(self.c, self.n)))
        self.cn_fine = 0.5*(fd.dot(self.c_fine, self.n_fine) + abs(fd.dot(self.c_fine, self.n_fine)))  # noqa

        # PDE problem RHS on coarse & fine mesh
        self.a = self.phi*self.du_trial*fd.dx(domain=self.mesh)
        self.a_fine = self.phi_fine*self.du_trial_fine*fd.dx(domain=self.mesh_fine)  # noqa

        # PDE problem LHS on coarse & fine mesh
        #       on coarse mesh
        self.L1 = self.dtc*(self.u*fd.div(self.phi*self.c)*fd.dx(domain=self.mesh)  # noqa
                - fd.conditional(fd.dot(self.c, self.n) < 0, self.phi*fd.dot(self.c, self.n)*self.u_in, 0.0)*fd.ds(domain=self.mesh)  # noqa
                - fd.conditional(fd.dot(self.c, self.n) > 0, self.phi*fd.dot(self.c, self.n)*self.u, 0.0)*fd.ds(domain=self.mesh)  # noqa
                - (self.phi('+') - self.phi('-'))*(self.cn('+')*self.u('+') - self.cn('-')*self.u('-'))*fd.dS(domain=self.mesh))  # noqa
        self.L2 = fd.replace(self.L1, {self.u: self.u1})
        self.L3 = fd.replace(self.L1, {self.u: self.u2})
        #       on fine mesh
        self.L1_fine = self.dtc*(self.u_fine*fd.div(self.phi_fine*self.c_fine)*fd.dx(domain=self.mesh_fine)  # noqa
                - fd.conditional(fd.dot(self.c_fine, self.n_fine) < 0, self.phi_fine*fd.dot(self.c_fine, self.n_fine)*self.u_in_fine, 0.0)*fd.ds(domain=self.mesh_fine)  # noqa
                - fd.conditional(fd.dot(self.c_fine, self.n_fine) > 0, self.phi_fine*fd.dot(self.c_fine, self.n_fine)*self.u_fine, 0.0)*fd.ds(domain=self.mesh_fine)  # noqa
                - (self.phi_fine('+') - self.phi_fine('-'))*(self.cn_fine('+')*self.u_fine('+') - self.cn_fine('-')*self.u_fine('-'))*fd.dS(domain=self.mesh_fine))  # noqa
        self.L2_fine = fd.replace(self.L1_fine, {self.u_fine: self.u1_fine})
        self.L3_fine = fd.replace(self.L1_fine, {self.u_fine: self.u2_fine})

        # vars for storing final solutions
        self.du = fd.Function(self.scalar_space)
        self.du_fine = fd.Function(self.scalar_space_fine)

        # PDE solver (one coarse & fine mesh) setup:
        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}  # noqa
        #       On coarse mesh
        self.prob1 = fd.LinearVariationalProblem(self.a, self.L1, self.du)
        self.solv1 = fd.LinearVariationalSolver(self.prob1, solver_parameters=params)  # noqa
        self.prob2 = fd.LinearVariationalProblem(self.a, self.L2, self.du)
        self.solv2 = fd.LinearVariationalSolver(self.prob2, solver_parameters=params)  # noqa
        self.prob3 = fd.LinearVariationalProblem(self.a, self.L3, self.du)
        self.solv3 = fd.LinearVariationalSolver(self.prob3, solver_parameters=params)  # noqa
        #       On fine mesh
        self.prob1_fine = fd.LinearVariationalProblem(self.a_fine, self.L1_fine, self.du_fine)  # noqa
        self.solv1_fine = fd.LinearVariationalSolver(self.prob1_fine, solver_parameters=params)  # noqa
        self.prob2_fine = fd.LinearVariationalProblem(self.a_fine, self.L2_fine, self.du_fine)  # noqa
        self.solv2_fine = fd.LinearVariationalSolver(self.prob2_fine, solver_parameters=params)  # noqa
        self.prob3_fine = fd.LinearVariationalProblem(self.a_fine, self.L3_fine, self.du_fine)  # noqa
        self.solv3_fine = fd.LinearVariationalSolver(self.prob3_fine, solver_parameters=params)  # noqa

    def solve_u(self, t):
        """
        Solve the PDE problem using RK (SSPRK) scheme on the coarse mesh
        store the solution field to a varaible: self.u_cur
        """
        c_exp = wm.get_c(self.x, self.y, t, alpha=self.alpha)
        c_temp = fd.Function(self.vector_space).interpolate(c_exp)
        self.c.project(c_temp)

        self.solv1.solve()
        self.u1.assign(self.u + self.du)

        self.solv2.solve()
        self.u2.assign(0.75*self.u + 0.25*(self.u1 + self.du))

        self.solv3.solve()
        self.u_cur.assign((1.0/3.0)*self.u + (2.0/3.0)*(self.u2 + self.du))

    def solve_u_fine(self, t):
        """
        Solve the PDE problem using RK (SSPRK) scheme on the fine mesh
        store the solution field to a varaible: self.u_cur_fine
        """
        c_exp = wm.get_c(self.x_fine, self.y_fine, t, alpha=self.alpha)
        c_temp = fd.Function(self.vector_space_fine).interpolate(c_exp)
        self.c_fine.project(c_temp)

        self.solv1_fine.solve()
        self.u1_fine.assign(self.u_fine + self.du_fine)

        self.solv2_fine.solve()
        self.u2_fine.assign(0.75*self.u_fine + 0.25*(self.u1_fine + self.du_fine))  # noqa

        self.solv3_fine.solve()
        self.u_cur_fine.assign((1.0/3.0)*self.u_fine + (2.0/3.0)*(self.u2_fine + self.du_fine))  # noqa

    def project_u_(self):
        self.u.project(self.u_fine_buffer)
        return

    def make_log_dir(self):
        wm.mkdir_if_not_exist(self.log_path)

    def make_plot_dir(self):
        wm.mkdir_if_not_exist(self.plot_path)

    def make_plot_more_dir(self):
        wm.mkdir_if_not_exist(self.plot_more_path)

    def eval_problem(self):
        print("In eval problem")
        self.t = 0.0
        step = 0
        idx = 0
        res = {
            "deform_loss": None,                    # nodal position loss
            "tangled_element": None,                # tangled elements on a mesh  # noqa
            "error_og": None,                       # PDE error on original uniform mesh  # noqa
            "error_model": None,                    # PDE error on model generated mesh   # noqa
            "error_ma": None,                       # PDE error on MA generated mesh      # noqa
            "error_reduction_MA": None,             # PDE error reduced by using MA mesh  # noqa
            "error_reduction_model": None,          # PDE error reduced by using model mesh  # noqa
            "time_consumption_model": None,         # time consumed generating mesh inferenced by the model  # noqa
            "time_consumption_MA": None,            # time consumed generating mesh by Monge-Ampere method  # noqa
            "acceration_ratio": None,               # time_consumption_ma / time_consumption_model  # noqa
        }
        for i in range(self.n_step):
            print("evalutation, time: ", self.t)
            # data loading from raw file
            raw_data_path = self.dataset.file_names[idx]
            raw_data = np.load(raw_data_path, allow_pickle=True).item()
            data_t = raw_data.get('swirl_params')['t']
            y = raw_data.get('y')
            # error tracking lists init
            self.error_adapt_list = []
            self.error_og_list = []
            sample = next(iter(
                    DataLoader([self.dataset[idx]],
                               batch_size=1,
                               shuffle=False)))
            # solve PDE problem on fine mesh
            self.solve_u_fine(self.t)
            if (abs(self.t - data_t) < 1e-5):
                print(f"---- evaluating samples: step: {step}, t: {self.t:.5f}, data_t: {data_t:.5f}")  # noqa
                # Evaluation time step hit
                # initiate model inferencing ...
                self.model.eval()
                bs = 1
                with torch.no_grad():
                    start = time.perf_counter()

                    mesh_query_x = sample.mesh_feat[:, 0].view(-1, 1).detach().clone()
                    mesh_query_y = sample.mesh_feat[:, 1].view(-1, 1).detach().clone()
                    mesh_query = torch.cat([mesh_query_x, mesh_query_y], dim=-1)

                    coord_ori_x = sample.mesh_feat[:, 0].view(-1, 1)
                    coord_ori_y = sample.mesh_feat[:, 1].view(-1, 1)
                    coord_ori = torch.cat([coord_ori_x, coord_ori_y], dim=-1)

                    num_nodes = coord_ori.shape[-2] // bs
                    input_q, input_kv = generate_samples(bs=bs, num_samples_per_mesh=num_nodes, data=sample, device=self.device)
                    
                    (out, model_raw_output, out_monitor), (phix, phiy) = self.model(sample, input_q, input_kv, mesh_query)
                    end = time.perf_counter()
                    dur_ms = (end - start) * 1000

                # calculate solution on original mesh
                self.mesh.coordinates.dat.data[:] = self.init_coord
                self.project_u_()
                self.solve_u(self.t)
                function_space = fd.FunctionSpace(self.mesh, "CG", 1)
                self.uh = fd.Function(function_space).project(self.u_cur)

                # calculate solution on adapted mesh
                self.adapt_coord = y
                self.mesh.coordinates.dat.data[:] = self.adapt_coord
                self.mesh_new.coordinates.dat.data[:] = self.adapt_coord
                self.project_u_()
                self.solve_u(self.t)
                function_space_new = fd.FunctionSpace(self.mesh_new, "CG", 1)  # noqa
                self.uh_new = fd.Function(function_space_new).project(self.u_cur)  # noqa

                # error measuring
                error_og, error_adapt = self.get_error()
                print(
                    f"error_og: {error_og}, \terror_adapt: {error_adapt}"
                )

                res["error_og"] = error_og
                res["error_ma"] = error_adapt

                print("inspect out type: ", type(out.detach().cpu().numpy()))

                # check mesh integrity - Only perform evaluation on non-tangling mesh  # noqa
                num_tangle = wm.get_sample_tangle(out, sample.x[:, :2], sample.face)  # noqa
                if isinstance(num_tangle, torch.Tensor):
                    num_tangle = num_tangle.item()
                if (num_tangle > 0):  # has tangled elems:
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
                    os.path.join(self.ds_root, 'log'), idx
                )["time"]
                print(res)

                # metric calculation
                res["deform_loss"] = 1000 * torch.nn.L1Loss()(out, sample.y).item()
                res["time_consumption_model"] = dur_ms

                res["acceration_ratio"] = res["time_consumption_MA"] / res["time_consumption_model"]     # noqa
                res["error_reduction_MA"] = (res["error_og"] - res["error_ma"]) / res["error_og"]        # noqa
                res["error_reduction_model"] = (res["error_og"] - res["error_model"]) / res["error_og"]  # noqa

                # save file
                df = pd.DataFrame(res, index=[0])
                df.to_csv(os.path.join(self.log_path, f"log_{idx:04d}.csv"))
                # plot compare mesh
                plot_fig = wm.plot_mesh_compare_benchmark(
                    out.detach().cpu().numpy(), sample.y, sample.face, 
                    deform_loss=res["deform_loss"],
                    pde_loss_model=res["error_model"],
                    pde_loss_reduction_model=res["error_reduction_model"],
                    pde_loss_MA=res["error_ma"],
                    pde_loss_reduction_MA=res["error_reduction_MA"],
                    tangle=res["tangled_element"]
                )
                plot_fig.savefig(os.path.join(self.plot_path, f"plot_{idx:04d}.png"))  # noqa

                # more detailed plot - 3d plot and 2d plot with mesh
                fig = plt.figure(figsize=(8, 8))

                # 3D plot of MA solution
                ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                ax1.set_title('MA Solution (3D)')
                fd.trisurf(self.uh_new, axes=ax1)

                # 3D plot of model solution
                if (num_tangle == 0):
                    self.adapt_coord = out.detach().cpu().numpy()
                    self.mesh.coordinates.dat.data[:] = self.adapt_coord
                    self.mesh_new.coordinates.dat.data[:] = self.adapt_coord
                    self.project_u_()
                    self.solve_u(self.t)
                    function_space_new = fd.FunctionSpace(self.mesh_new, "CG", 1)  # noqa
                    uh_model = fd.Function(function_space_new).project(self.u_cur)  # noqa
                    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
                    ax2.set_title('Model Solution (3D)')
                    fd.trisurf(uh_model, axes=ax2)

                    # 2d plot and mesh for Model
                    ax4 = fig.add_subplot(2, 2, 4)
                    ax4.set_title('Soultion on Model mesh')
                    fd.tripcolor(
                        uh_model, cmap='coolwarm', axes=ax4)
                    self.mesh_new.coordinates.dat.data[:] = out.detach().cpu().numpy()  # noqa
                    fd.triplot(self.mesh_new, axes=ax4)

                # 2d plot and mesh for MA
                ax3 = fig.add_subplot(2, 2, 3)
                ax3.set_title('Soultion on MA mesh')
                fd.tripcolor(
                    self.uh_new, cmap='coolwarm', axes=ax3)
                self.mesh_new.coordinates.dat.data[:] = sample.y
                fd.triplot(self.mesh_new, axes=ax3)

                fig.savefig(os.path.join(self.plot_more_path, f"plot_{idx}.png"))  # noqa

                # plotting (visulisation during sovling)
                plot = False
                if plot is True:
                    self.plot_res()
                    plt.show()
                idx += 1
                plt.close()

            # time stepping and prep for next solving iter
            self.t += self.dt
            step += 1
            self.u_fine.assign(self.u_cur_fine)
            self.u_fine_buffer.assign(self.u_cur_fine)
        
        return

    def vis_evaluate(self, sample):
        """
        It would be great if we have some visuals here to assist
        out judgment.
        """
        print("In evaluation VISUALISATION")
        self.mesh.coordinates.dat.data[:] = sample.y
        fd.triplot(self.mesh)
        self.mesh.coordinates.dat.data[:] = self.init_coord
        plt.show()
        return

    def get_error(self):
        # solve on fine mesh
        function_space_fine = fd.FunctionSpace(self.mesh_fine, "CG", 1)
        self.solve_u_fine(self.t)
        u_fine = fd.Function(function_space_fine).project(self.u_cur_fine)  # noqa

        # solve on coarse mesh
        self.mesh.coordinates.dat.data[:] = self.init_coord
        self.project_u_()
        self.solve_u(self.t)
        u_og_2_fine = fd.project(self.u_cur, function_space_fine)

        # solve on coarse adapt mesh
        self.mesh.coordinates.dat.data[:] = self.adapt_coord
        self.project_u_()
        self.solve_u(self.t)
        u_adapt_2_fine = fd.project(self.u_cur, function_space_fine)

        # error calculation
        error_og = fd.errornorm(
            u_fine, u_og_2_fine, norm_type="L2"
        )
        error_adapt = fd.errornorm(
            u_fine,  u_adapt_2_fine, norm_type="L2"
        )

        # put mesh to init state
        self.mesh.coordinates.dat.data[:] = self.init_coord

        return error_og, error_adapt

    def plot_res(self):
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.set_title("Solution on fine mesh")
        fd.trisurf(self.u_cur_fine, axes=ax1)

        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        ax2.set_title("Solution on original mesh")
        fd.trisurf(self.uh, axes=ax2)

        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        ax3.set_title("Solution on adapt mesh")
        fd.trisurf(self.uh_new, axes=ax3)

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title("Fine mesh")
        fd.triplot(self.mesh_fine, axes=ax4)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title("Orignal mesh")
        fd.tripcolor(self.uh, axes=ax5, cmap='coolwarm')
        fd.triplot(self.mesh, axes=ax5)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title("adapted mesh")
        fd.tripcolor(self.uh_new, axes=ax6, cmap='coolwarm')
        fd.triplot(self.mesh_new, axes=ax6)

        return fig
