"""
Module for generating random exact solutions.
"""

import random

import firedrake as fd

__all__ = ["RandomSolutionGenerator"]


class RandomSolutionGenerator:
    r"""
    Class for generating a random solution field for a PDE with a scalar-valued solution.

    Random values are sampled from a Gaussian distribution.

    There are two forms of the solution that can be specified using the `use_iso`
    parameter. If `use_iso` is True, a simpler isotropic form of the solution is used,
    according to the formula

    ..math::
        u_{exact} += \exp\left(-1 \cdot \left(\frac{(x - \mu_x)^2 + (y - \mu_y)^2}{w}\right)\right)

    If `use_iso` is False, a more complex anisotropic form of the solution is used:

    ..math::
        u_{exact} += z \cdot \exp\left(-1 \cdot \left(\frac{(x - \mu_x)^2}{\sigma_x^2} + \frac{(y - \mu_y)^2}{\sigma_y^2}\right)\right)
    """

    def __init__(
        self,
        use_iso=False,
        dist_params={
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
            "sigma_mean_scaler": 1 / 4,
            "sigma_sigma_scaler": 1 / 6,
            "sigma_eps": 1 / 8,
        },
    ):
        """
        :kwarg use_iso: Flag for using simpler form of u
        :type use_iso: bool
        :kwarg dist_params: Parameters for Gaussian distribution.
        :type dist_params: dict
        :key max_dist: Maximum number of distributions
        :key n_dist: Number of distributions (if None, then random)
        :key x_start: Start of x range
        :key x_end: End of x range
        :key y_start: Start of y range
        :key y_end: End of y range
        :key z_max: Maximum value for z
        :key z_min: Minimum value for z
        :key w_min: Minimum value for w
        :key w_max: Maximum value for w
        :key c_min: Minimum value for c
        :key c_max: Maximum value for c
        :key sigma_mean_scaler: Scaler for sigma mean
        :key sigma_sigma_scaler: Scaler for sigma sigma
        :key sigma_eps: Epsilon value for sigma
        """
        # TODO: Pass params directly
        self.use_iso = use_iso
        self.dist_params = dist_params
        self.σ_dict = {"x": [], "y": []}
        self.μ_dict = {"x": [], "y": []}
        self.z_list = []
        self.w_list = []
        self.set_dist_params(eps=self.dist_params["sigma_eps"])

        self.u_exact = 0  # analytical solution
        self.f = 0  # simulated source function # TODO: Never gets used?
        self.function_space = None  # TODO: Never gets used?
        self.LHS = None  # TODO: Never gets used?
        self.RHS = None  # TODO: Never gets used?
        self.bc = None  # boundary conditions # TODO: Never gets used?

    def set_dist_params(self, eps=1 / 20):
        """
        Set parameters for Gaussian distribution from dist_params.

        :param eps: Epsilon value to ensure minimum standard deviation.
        :type eps: float
        """
        if self.dist_params["n_dist"] is None:
            self.n_dist = random.randint(1, self.dist_params["max_dist"])
        else:
            self.n_dist = self.dist_params["n_dist"]
        print("Generating {} Gaussian distributions".format(self.n_dist))
        for i in range(self.n_dist):
            σ_mean = (
                self.dist_params["x_end"] - self.dist_params["x_start"]
            ) * self.dist_params["sigma_mean_scaler"]
            σ_sigma = (
                self.dist_params["x_end"] - self.dist_params["x_start"]
            ) * self.dist_params["sigma_sigma_scaler"]

            self.μ_dict["x"].append(
                round(
                    random.uniform(
                        self.dist_params["c_min"], self.dist_params["c_max"]
                    ),
                    3,
                )
            )  # noqa
            self.μ_dict["y"].append(
                round(
                    random.uniform(
                        self.dist_params["c_min"], self.dist_params["c_max"]
                    ),
                    3,
                )
            )  # noqa

            self.σ_dict["x"].append(max(round(random.gauss(σ_mean, σ_sigma), 3), eps))
            self.σ_dict["y"].append(max(round(random.gauss(σ_mean, σ_sigma), 3), eps))
            self.z_list.append(
                round(
                    random.uniform(
                        self.dist_params["z_min"], self.dist_params["z_max"]
                    ),
                    3,
                )
            )
            self.w_list.append(
                round(
                    random.uniform(
                        self.dist_params["w_min"], self.dist_params["w_max"]
                    ),
                    3,
                )
            )

    def get_dist_params(self):
        """
        :return: Distribution parameters
        :rtype: dict
        """
        # TODO: Why does this set differ from the one passed in?
        # TODO: Create class for distribution parameters
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

    def get_u_exact(
        self,
        params={
            "x": None,
            "y": None,
            "V": None,
            "u": None,
            "v": None,
        },
    ):
        """
        :return: analytical solution field
        :rtype: `firedrake.firedrake.Function`
        """
        # TODO: Pass params directly
        # TODO: Docstring arg descriptions
        x, y = (params[key] for key in ("x", "y"))
        self.u_exact = 0
        if self.use_iso:  # use simpler form of u
            for i in range(self.n_dist):
                μ_x = self.μ_dict["x"][i]
                μ_y = self.μ_dict["y"][i]
                w = self.w_list[i]
                self.u_exact += fd.exp(-1 * ((((x - μ_x) ** 2) + ((y - μ_y) ** 2)) / w))
        else:  # use more complex form of u
            for i in range(self.n_dist):
                σ_x = self.σ_dict["x"][i]
                σ_y = self.σ_dict["y"][i]
                μ_x = self.μ_dict["x"][i]
                μ_y = self.μ_dict["y"][i]
                z = self.z_list[i]
                self.u_exact += z * fd.exp(
                    -1 * ((((x - μ_x) ** 2) / (σ_x**2)) + (((y - μ_y) ** 2) / (σ_y**2)))
                )
        return self.u_exact
