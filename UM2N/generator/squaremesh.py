from firedrake.mesh import Mesh
import gmsh
import os

__all__ = ["UnstructuredUnitSquareMesh"]


class UnstructuredUnitSquareMesh:
    """
    Generate an unstructured mesh of a 2D square domain using Gmsh.
    """

    def __init__(self, mesh_type=2):
        """
        :kwarg mesh_type: Gmsh algorithm number
        :type mesh_type: int
        """
        # TODO: More detail on Gmsh algorithm number
        self.mesh_type = mesh_type

        self.points = []
        self.lines = []
        self._mesh = None

    def generate_mesh(self, res=1e-1, file_path="./temp.msh", remove_file=False):
        """
        Generate a mesh at a given resolution level.

        :kwarg res: mesh resolution (element diameter)
        :type res: float
        :kwarg file_path: file name for saving the mesh in .msh format
        :type file_path: str
        :kwarg remove_file: should the file be removed after generation?
        :type remove_file: bool
        """
        gmsh.initialize()
        gmsh.model.add("t1")
        # temp vars
        self.points = []
        self.lines = []
        # generate mesh
        self.corners = ((0, 0), (1, 0), (1, 1), (0, 1))
        self.points = [
            gmsh.model.geo.addPoint(*corner, 0, res) for corner in self.corners
        ]
        self.get_line()
        self.get_curve()
        self.get_plane()
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", self.mesh_type)
        self.get_boundaries()
        gmsh.model.addPhysicalGroup(2, [1], name="My surface")
        gmsh.model.mesh.generate(2)
        gmsh.write(file_path)
        gmsh.finalize()
        self.num_boundary = len(self.lines)
        self._mesh = Mesh(file_path)
        if remove_file:
            os.remove(file_path)
        return self._mesh

    def get_line(self):
        for point, point_next in zip(self.points, self.points[1:] + [self.points[0]]):
            self.lines.append(gmsh.model.geo.addLine(point, point_next))

    def get_boundaries(self):
        for i, line_tag in enumerate(self.lines):
            gmsh.model.addPhysicalGroup(1, [line_tag], i + 1)
            gmsh.model.setPhysicalName(1, i + 1, "Boundary " + str(i + 1))

    def get_curve(self):
        gmsh.model.geo.addCurveLoop([i + 1 for i in range(len(self.points))], 1)

    def get_plane(self):
        gmsh.model.geo.addPlaneSurface([1], 1)

    def load_mesh(self, file_path):
        self._mesh = Mesh(file_path)
        return self._mesh
