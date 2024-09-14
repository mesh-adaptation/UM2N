# Author: Chunyang Wang
# GitHub Username: chunyang-w

import os
import random
import firedrake as fd

os.environ["OMP_NUM_THREADS"] = "1"
__all__ = ["EquationSolver"]
random.seed(42)


class EquationSolver:
    def __init__(
        self, params={"LHS": None, "RHS": None, "function_space": None, "bc": None}
    ):
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
        fd.solve(
            self.LHS == self.RHS,
            uh,
            solver_parameters={"ksp_type": "cg", "pc_type": "none"},
            bcs=self.bc,
        )
        self.uh = uh
        return self.uh
