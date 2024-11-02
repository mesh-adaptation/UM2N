"""
Module for handling generating unstructured meshes.
"""

from firedrake.mesh import Mesh

import abc
import gmsh
import numpy as np
import os
import random

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
        :kwarg scale: overall scale factor
        :type scale: float
        :kwarg mesh_type: Gmsh algorithm number
        :type mesh_type: int
        """
        self.scale = scale
        # TODO: More detail on Gmsh algorithm number
        self.mesh_type = mesh_type
        self._mesh = None

    @property
    @abc.abstractmethod
    def corners(self):
        """
        :returns: corner vertices of the domain
        :rtype: tuple
        """
        pass

    def generate_mesh(self, res=1e-1, file_path="./temp.msh", remove_file=False):
        """
        Generate a mesh at a given resolution level.

        :kwarg res: mesh resolution (element diameter)
        :type res: float
        :kwarg file_path: file name for saving the mesh in .msh format
        :type file_path: str
        :kwarg remove_file: should the .msh file be removed after generation? (False by
            default)
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
        gmsh.write(file_path)
        gmsh.finalize()
        self.num_boundary = len(self._lines)
        self._mesh = Mesh(file_path)
        if remove_file:
            os.remove(file_path)
        return self._mesh

    def load_mesh(self, file_path):
        """
        Load a mesh from a file saved in .msh format.

        :arg file_path: filename including the .msh extension
        :type file_path: str
        :returns: mesh loaded from file
        :rtype: :class:`firedrake.mesh.MeshGeometry`
        """
        self._mesh = Mesh(file_path)
        return self._mesh


class UnstructuredSquareMeshGenerator(UnstructuredMeshGenerator):
    """
    Generate an unstructured mesh of a 2D square domain using Gmsh.
    """

    @property
    def corners(self):
        """
        :returns: corner vertices of the square domain
        :rtype: tuple
        """
        return ((0, 0), (self.scale, 0), (self.scale, self.scale), (0, self.scale))


class UnstructuredRandomPolygonalMeshGenerator(UnstructuredMeshGenerator):
    """
    Create a random polygonal mesh by spliting the edge of a
    square randomly.
    """

    def __init__(self, scale=1.0, mesh_type=2):
        """
        :kwarg scale: overall scale factor
        :type scale: float
        :kwarg mesh_type: Gmsh algorithm number
        :type mesh_type: int
        """
        super().__init__(mesh_type=mesh_type)
        self.scale = scale

    @staticmethod
    def sample_uniform(mean, interval):
        """
        Sample a point from a uniform distribution with a given mean and interval.

        Note that the interval is *either side* of the mean, not the overall interval.

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
        :returns: corner vertices of a randomly generated polygonal domain
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
