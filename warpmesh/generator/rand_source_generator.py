# Author: Chunyang Wang
# GitHub Username: chunyang-w

import random
import firedrake as fd

__all__ = ["RandSourceGenerator"]


class RandSourceGenerator():
    """
    Class for generating a random Helmholtz equation based on a
    Gaussian distribution.

    Attributes:
        simple_u (bool): Flag for using simpler form of u.
        dist_params (dict): Parameters for Gaussian distribution.
        u_exact: Analytical Helmholtz equation solution.
        f: Simulated source function.
        function_space: Function space for problem.
        LHS: Left-hand side of Helmholtz equation.
        RHS: Right-hand side of Helmholtz equation.
        bc: Dirichlet boundary condition.
    """

    def __init__(self, use_iso=False, dist_params={
        "max_dist": 10,
        "n_dist": None,  # if None, then random
        "x_start": 0,
        "x_end": 1,
        "y_start": 0,
        "y_end": 1,
        "z_max": 1,
        "z_min": 0,
        "w_min": 0.05,
        "w_max": 0.2,
        "c_min": 0.2,
        "c_max": 0.8,
    }):
        """
        Initialize RandomHelmholtzGenerator.

        Parameters:
            simple_u (bool): Use simpler form of u (isotripic dataset) if True.
              Default False.
            dist_params (dict): Parameters for Gaussian distribution.
        """
        self.use_iso = use_iso
        self.dist_params = dist_params
        self.σ_dict = {
            "x": [],
            "y": []
        }
        self.μ_dict = {
            "x": [],
            "y": []
        }
        self.z_list = []
        self.w_list = []
        self.set_dist_params()

        self.u_exact = 0  # analytical solution
        self.f = 0  # simulated source function
        self.function_space = None
        self.LHS = None
        self.RHS = None
        self.bc = None  # boundary conditions

    def set_dist_params(self, eps=1/8):
        """
        Set parameters for Gaussian distribution from dist_params.
        """
        if (self.dist_params["n_dist"] is None):
            self.n_dist = random.randint(1, self.dist_params["max_dist"])
        else:
            self.n_dist = self.dist_params["n_dist"]
        print("Generating {} Gaussian distributions".format(self.n_dist))
        for i in range(self.n_dist):
            σ_mean = (
                self.dist_params["x_end"] - self.dist_params["x_start"]) / 4
            σ_sigma = (
                self.dist_params["x_end"] - self.dist_params["x_start"]) / 6

            self.μ_dict["x"].append(round(random.uniform(
                self.dist_params["c_min"], self.dist_params["c_max"]), 3)) # noqa
            self.μ_dict["y"].append(round(random.uniform(
                self.dist_params["c_min"], self.dist_params["c_max"]), 3)) # noqa

            self.σ_dict["x"].append(
                max(round(random.gauss(σ_mean, σ_sigma), 3), eps))
            self.σ_dict["y"].append(
                max(round(random.gauss(σ_mean, σ_sigma), 3), eps))
            self.z_list.append(round(random.uniform(
                self.dist_params["z_min"],
                self.dist_params["z_max"]), 3))
            self.w_list.append(round(random.uniform(
                self.dist_params["w_min"],
                self.dist_params["w_max"]), 3))

    def get_dist_params(self):
        """
        Return dictionary containing distribution parameters.

        Returns:
            dict: Dictionary of distribution parameters.
        """
        dist = {
            "n_dist": self.n_dist,
            "σ_x": self.σ_dict["x"],
            "σ_y": self.σ_dict["y"],
            "μ_x": self.μ_dict["x"],
            "μ_y": self.μ_dict["y"],
            "z": self.z_list,
            "w": self.w_list,
            "use_iso": self.use_iso,
        }
        return dist

    def get_u_exact(self, params={
            "x": None,
            "y": None,
            "V": None,
            "u": None,
            "v": None,
    }):
        """
        Return analytical solution field.
        Returns:
            firedrake.Function: Analytical solution.
        """
        x, y = (
            params[key] for key in ("x", "y"))
        self.u_exact = 0
        if (self.use_iso):  # use simpler form of u
            for i in range(self.n_dist):
                μ_x = self.μ_dict["x"][i]
                μ_y = self.μ_dict["y"][i]
                w = self.w_list[i]
                self.u_exact += fd.exp(-1 * (
                    (((x-μ_x)**2) + ((y-μ_y)**2)) / w
                ))
        else:  # use more complex form of u
            for i in range(self.n_dist):
                σ_x = self.σ_dict["x"][i]
                σ_y = self.σ_dict["y"][i]
                μ_x = self.μ_dict["x"][i]
                μ_y = self.μ_dict["y"][i]
                z = self.z_list[i]
                self.u_exact += z * fd.exp(-1 * (
                    (((x-μ_x)**2) / (σ_x**2)) +
                    (((y-μ_y)**2) / (σ_y**2))
                ))
        return self.u_exact
