# Author: Chunyang Wang
# GitHub Username: acse-cw1722
import os
import random
import firedrake as fd
import movement as mv
# import matplotlib.pyplot as plt

os.environ['OMP_NUM_THREADS'] = "1"
__all__ = ["MeshGenerator", "HelmholtzSolver",
           "RandomHelmholtzGenerator", "HelmholtzGenerator"]
random.seed(42)


class MeshGenerator():
    """
    Responsible for generating and moving a mesh based on a given Helmholtz
     equation.
    This method is based on Pyroteus/movement.

    Attributes:
    - eq: The Helmholtz equation object.
    - num_grid_x: Number of grid points in the x-dimension.
    - num_grid_y: Number of grid points in the y-dimension.
    - mesh: The initial mesh.
    """
    def __init__(self, params={
        "num_grid_x": None,
        "num_grid_y": None,
        "helmholtz_eq": None,
        "mesh": None,
    }):
        self.eq = params["helmholtz_eq"]
        self.num_grid_x = params["num_grid_x"]
        self.num_grid_y = params["num_grid_y"]
        self.mesh = params["mesh"]

    def move_mesh(self):
        """
        Moves the mesh using the Monge-Ampere equation.
        Computes and stores the Jacobian and its determinant.

        Returns:
        - The moved mesh
        """
        mover = mv.MongeAmpereMover(
            self.mesh, self.monitor_func, method="relaxation", rtol=1e-3)
        mover.move()
        # extract Hessian of the movement
        sigma = mover.sigma
        I = fd.Identity(2)  # noqa
        jacobian = I + sigma
        jacobian_det = fd.Function(
            self.eq.function_space, name="jacobian_det")
        jacobian_det.project(
            jacobian[0, 0] * jacobian[1, 1] -
            jacobian[0, 1] * jacobian[1, 0])
        self.jacob_det = jacobian_det
        self.jacob = jacobian
        return self.mesh

    def get_monitor_val(self):
        """
        Returns the monitor function value used for mesh movement.
        """
        return self.monitor_val

    def get_jacobian(self):
        """
        Returns the Jacobian of the mesh movement.
        """
        return self.jacob

    def get_jacobian_det(self):
        """
        Returns the determinant of the Jacobian of the mesh movement.
        """
        return self.jacob_det

    def get_hessian(self, mesh):
        """
        Computes and returns the Hessian of the Helmholtz equation on the
        given mesh.

        Parameters:
        - mesh: The mesh on which to compute the Hessian.

        Returns:
        - The Hessian as a projection in the function space.
        """
        res = self.eq.discretise(mesh)
        function_space_ten = fd.TensorFunctionSpace(mesh, "CG", 1)

        solver = HelmholtzSolver(params={
            "LHS": res["LHS"],
            "RHS": res["RHS"],
            "function_space": res["function_space"],
            "bc": res["bc"],
        })
        uh = solver.solve_eq()

        n = fd.FacetNormal(mesh)
        l2_projection = fd.Function(function_space_ten)
        H, h = fd.TrialFunction(
            function_space_ten), fd.TestFunction(function_space_ten)
        a = fd.inner(h, H) * fd.dx(domain=mesh)
        L = -fd.inner(fd.div(h), fd.grad(uh)) * fd.dx(domain=mesh)
        L += fd.dot(fd.grad(uh), fd.dot(h, n)) * fd.ds(domain=mesh)
        prob = fd.LinearVariationalProblem(a, L, l2_projection)
        hessian_prob = fd.LinearVariationalSolver(prob)
        hessian_prob.solve()
        return l2_projection

    def monitor_func(self, mesh):
        """
        Computes the monitor function value based on the Hessian of the
        Helmholtz equation.

        Parameters:
        - mesh: The mesh on which to compute the monitor function.

        Returns:
        - The monitor function value.
        """
        res = self.eq.discretise(mesh)
        function_space = res["function_space"]
        hessian_norm = fd.Function(function_space)
        l2_projection = self.get_hessian(mesh)
        hessian_norm.project(
            l2_projection[0, 0] ** 2 +
            l2_projection[0, 1] ** 2 +
            l2_projection[1, 0] ** 2 +
            l2_projection[1, 1] ** 2)
        hessian_norm /= hessian_norm.vector().max()
        monitor_val = 1 + 5 * hessian_norm
        return monitor_val


class HelmholtzSolver():
    def __init__(self, params={
        "LHS": None,
        "RHS": None,
        "function_space": None,
        "bc": None
    }):
        self.uh = None
        self.function_space = params["function_space"]
        self.LHS = params["LHS"]
        self.RHS = params["RHS"]
        self.bc = params["bc"]

    def solve_eq(self):
        """
        Solves the Helmholtz equation
        """
        uh = fd.Function(self.function_space)
        fd.solve(self.LHS == self.RHS, uh, solver_parameters={
            'ksp_type': 'cg', 'pc_type': 'none'}, bcs=self.bc)
        self.uh = uh
        return self.uh


class RandomHelmholtzGenerator():
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

    def __init__(self, simple_u=False, dist_params={
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
    }):
        """
        Initialize RandomHelmholtzGenerator.

        Parameters:
            simple_u (bool): Use simpler form of u (isotripic dataset) if True.
              Default False.
            dist_params (dict): Parameters for Gaussian distribution.
        """
        self.simple_u = simple_u
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

    def set_dist_params(self):
        """
        Set parameters for Gaussian distribution from dist_params.
        """
        if (self.dist_params["n_dist"] is None):
            self.n_dist = random.randint(1, self.dist_params["max_dist"])
        else:
            self.n_dist = self.dist_params["n_dist"]
        print("Generating {} Gaussian distributions".format(self.n_dist))
        for i in range(self.n_dist):
            σ_mean = random.gauss((
                self.dist_params["x_end"] -
                self.dist_params["x_start"])/24)
            σ_sigma = random.gauss((
                self.dist_params["y_end"] -
                self.dist_params["y_start"])/48)

            self.μ_dict["x"].append(round(random.uniform(
                self.dist_params["x_start"], self.dist_params["x_end"]), 4))
            self.μ_dict["y"].append(round(random.uniform(
                self.dist_params["y_start"], self.dist_params["y_end"]), 4))
            self.σ_dict["x"].append(round(random.gauss(σ_mean, σ_sigma), 4))
            self.σ_dict["y"].append(round(random.gauss(σ_mean, σ_sigma), 4))
            self.z_list.append(round(random.uniform(
                self.dist_params["z_min"],
                self.dist_params["z_max"]), 4))
            self.w_list.append(round(random.uniform(
                self.dist_params["w_min"],
                self.dist_params["w_max"]), 4))

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
            "simple_u": self.simple_u,
        }
        return dist

    def discretise(self, mesh=fd.UnitSquareMesh(10, 10)):
        """
        Discretize source distribution on given mesh.
        Constructs LHS, RHS, and bc for Helmholtz equation.

        Parameters:
            mesh: Mesh to discretize distribution.

        Returns:
            dict: Dictionary of mesh, function_space, u_exact,
                  LHS, RHS, bc, and f.
        """
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        self.u_exact = 0

        if (self.simple_u is not True):  # use more complex u
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
        else:  # use simpler form of u
            for i in range(self.n_dist):
                μ_x = self.μ_dict["x"][i]
                μ_y = self.μ_dict["y"][i]
                w = self.w_list[i]
                self.u_exact += fd.exp(-1 * (
                    (((x-μ_x)**2) + ((y-μ_y)**2)) / w
                ))

        self.f = -1 * fd.div(fd.grad(self.u_exact)) + self.u_exact
        self.LHS = (fd.dot(
            fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.function_space = V
        self.bc = fd.DirichletBC(
            self.function_space, self.u_exact, "on_boundary")

        return {
            "mesh": mesh,
            "function_space": self.function_space,
            "u_exact": self.u_exact,
            "LHS": self.LHS,
            "RHS": self.RHS,
            "bc": self.bc,
            "f": self.f,
        }


class HelmholtzGenerator():
    """
    Generates a given  Helmholtz equation.
    """
    def __init__(self, params={
        "f_func": None,
        "u_exact_func": None,
    }):
        self.u_exact_func = params["u_exact_func"]  # analytical solution
        self.f_func = params["f_func"]
        self.bc = None  # boundary conditions
        self.function_space = None
        self.LHS = None
        self.RHS = None
        self.bc = None  # boundary conditions

    def discretise(self, mesh=fd.UnitSquareMesh(10, 10)):
        """
        Discretises the source distribution to a certain mesh
        """
        x, y = fd.SpatialCoordinate(mesh)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        self.u_exact = self.u_exact_func(x, y)
        self.f = self.f_func(x, y)

        self.LHS = (fd.dot(
            fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
        self.RHS = self.f * v * fd.dx(domain=mesh)
        self.function_space = V
        self.bc = fd.DirichletBC(
            self.function_space, self.u_exact, "on_boundary")

        return {
            "mesh": mesh,
            "function_space": self.function_space,
            "u_exact": self.u_exact,
            "LHS": self.LHS,
            "RHS": self.RHS,
            "bc": self.bc,
            "f": self.f
        }
