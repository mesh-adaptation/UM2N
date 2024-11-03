"""
Module for handling generating unstructured meshes.
"""

import abc
import os
import random

import gmsh
import numpy as np
from firedrake.mesh import Mesh

__all__ = [
    "UnstructuredSquareMeshGenerator",
    "UnstructuredRandomPolygonalMeshGenerator",
]


class UnstructuredMeshGenerator(abc.ABC):
    """
    Base class for mesh generators.
    """

    def __init__(self, scale=1.0, mesh_type=2):
        """
        :kwarg scale: overall scale factor for the domain size (default: 1.0)
        :type scale: float
        :kwarg mesh_type: Gmsh algorithm number (default: 2)
        :type mesh_type: int
        """
        self.scale = scale
        # TODO: More detail on Gmsh algorithm number (#50)
        self.mesh_type = mesh_type
        self._mesh = None

    @property
    @abc.abstractmethod
    def corners(self):
        """
        Property defining the coordinates of the corner vertices of the domain to be
        meshed.

        :returns: coordinates of the corner vertices of the domain
        :rtype: tuple
        """
        pass

    def generate_mesh(self, res=0.1, output_filename="./temp.msh", remove_file=False):
        """
        Generate a mesh at a given resolution level.

        :kwarg res: mesh resolution (element diameter) (default: 0.1, suitable for mesh
            with scale 1.0)
        :type res: float
        :kwarg output_filename: filename for saving the mesh, including the path and .msh
            extension (default: './temp.msh')
        :type output_filename: str
        :kwarg remove_file: should the .msh file be removed after generation? (default:
            False)
        :type remove_file: bool
        :returns: mesh generated
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        gmsh.initialize()
        gmsh.model.add("t1")
        self.lc = res
        self._points = [
            gmsh.model.geo.addPoint(*corner, 0, self.lc) for corner in self.corners
        ]
        self._lines = [
            gmsh.model.geo.addLine(point, point_next)
            for point, point_next in zip(
                self._points, self._points[1:] + [self._points[0]]
            )
        ]
        gmsh.model.geo.addCurveLoop([i + 1 for i in range(len(self._points))], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", self.mesh_type)
        for i, line_tag in enumerate(self._lines):
            gmsh.model.addPhysicalGroup(1, [line_tag], i + 1)
            gmsh.model.setPhysicalName(1, i + 1, "Boundary " + str(i + 1))
        gmsh.model.addPhysicalGroup(2, [1], name="My surface")
        gmsh.model.mesh.generate(2)
        gmsh.write(output_filename)
        gmsh.finalize()
        self.num_boundary = len(self._lines)
        self._mesh = Mesh(output_filename)
        if remove_file:
            os.remove(output_filename)
        return self._mesh

    def load_mesh(self, filename):
        """
        Load a mesh from a file saved in .msh format.

        :arg filename: filename including path and the .msh extension
        :type filename: str
        :returns: mesh loaded from file
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        self._mesh = Mesh(filename)
        return self._mesh


class UnstructuredSquareMeshGenerator(UnstructuredMeshGenerator):
    """
    Generate an unstructured mesh of a 2D square domain using Gmsh.
    """

    @property
    def corners(self):
        """
        Property defining the coordinates of the corner vertices of the domain to be
        meshed.

        :returns: coordinates of the corner vertices of the domain
        :rtype: tuple
        """
        return ((0, 0), (self.scale, 0), (self.scale, self.scale), (0, self.scale))


class UnstructuredRandomPolygonalMeshGenerator(UnstructuredMeshGenerator):
    """
    Create a random polygonal mesh by spliting the edge of a
    square randomly.
    """

    def generate_mesh(self, seed=None, **kwargs):
        """
        Generate a mesh at a given resolution level.

        :kwarg res: mesh resolution (element diameter) (default: 0.1, suitable for mesh
            with scale 1.0)
        :type res: float
        :kwarg output_filename: filename for saving the mesh, including the path and .msh
            extension (default: './temp.msh')
        :type output_filename: str
        :kwarg remove_file: should the .msh file be removed after generation? (default:
            False)
        :type remove_file: bool
        :kwarg seed: optional random seed for reproducibility (default: None, i.e., do
            not use a random seed)
        :type seed: int
        :returns: mesh generated
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        # TODO: Write tests that make use of the random seed (#51)
        if seed is not None:
            np.random.seed(seed)
        return super().generate_mesh(**kwargs)

    @staticmethod
    def sample_uniform(mean, interval):
        """
        Sample a point from a uniform distribution with a given mean and interval.

        Note that the interval is *either side* of the mean, not the overall interval,
        i.e., overall interval is symmetric: `[mean - interval, mean + interval]`.

        :arg mean: the mean value of the uniform distribution
        :type mean: float
        :arg interval: the interval either side of the mean
        :type interval: float
        :returns: sampled point
        :rtype: float
        """
        return random.uniform(mean - interval, mean + interval)

    @property
    def corners(self):
        """
        Property defining the coordinates of the corner vertices of the domain to be
        meshed.

        :returns: coordinates of the corner vertices of the domain
        :rtype: tuple
        """
        if hasattr(self, "_corners"):
            return self._corners
        start = 0
        finish = self.scale
        split_threshold = 0.3
        mid = (start + finish) / 2
        quarter = (start + mid) / 2
        three_quarter = (mid + finish) / 2
        mid_interval = (finish - start) / 3
        quarter_interval = (mid - start) / 4
        points = []
        split_p = np.random.uniform(0, 1, 4)
        # edge 1
        if split_p[0] < split_threshold:
            points.append([self.sample_uniform(quarter, quarter_interval), 0])
            points.append([self.sample_uniform(three_quarter, quarter_interval), 0])
        else:
            points.append([self.sample_uniform(mid, mid_interval), 0])
        # edge 2
        if split_p[1] < split_threshold:
            points.append([self.scale, self.sample_uniform(quarter, quarter_interval)])
            points.append(
                [self.scale, self.sample_uniform(three_quarter, quarter_interval)]
            )
        else:
            points.append([self.scale, self.sample_uniform(mid, mid_interval)])
        # edge 3
        if split_p[2] < split_threshold:
            points.append(
                [self.sample_uniform(three_quarter, quarter_interval), self.scale]
            )
            points.append([self.sample_uniform(quarter, quarter_interval), self.scale])
        else:
            points.append([self.sample_uniform(mid, mid_interval), self.scale])
        # edge 4
        if split_p[3] < split_threshold:
            points.append([0, self.sample_uniform(three_quarter, quarter_interval)])
            points.append([0, self.sample_uniform(quarter, quarter_interval)])
        else:
            points.append([0, self.sample_uniform(mid, mid_interval)])
        self._corners = tuple(points)
        return self._corners
