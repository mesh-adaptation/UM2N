import firedrake as fd
import numpy as np
import gmsh
import random


__all__ = ["RandPolyMesh"]


class RandPolyMesh():
    """
    Create a random polygonal mesh by spliting the edge of a
    square randomly.
    """
    def __init__(self, scale=1.0):
        # params setup
        self.scale = scale
        self.start = 0
        self.end = self.scale
        self.split_threshold = 0.3
        self.mid = (self.start + self.end) / 2
        self.quater = (self.start + self.mid) / 2
        self.three_quater = (self.mid + self.end) / 2
        self.mid_interval = (self.end - self.start) / 3
        self.quater_interval = (self.mid - self.start) / 4
        # temp vars
        self.points = []
        self.lines = []
        # generate mesh
        self.get_rand_points()
        return

    def get_mesh(self, res=1e-1, file_path="./temp.msh"):
        gmsh.initialize()
        gmsh.model.add("t1")
        # params setup
        self.lc = res
        self.start = 0
        self.end = self.scale
        self.mid = (self.start + self.end) / 2
        self.quater = (self.start + self.mid) / 2
        self.three_quater = (self.mid + self.end) / 2
        self.mid_interval = (self.end - self.start) / 3
        self.quater_interval = (self.mid - self.start) / 4
        self.file_path = file_path
        # temp vars
        self.points = []
        self.lines = []
        # generate mesh
        self.get_points()
        self.get_line()
        self.get_curve()
        self.get_plane()
        gmsh.model.geo.synchronize()
        self.get_boundaries()
        gmsh.model.addPhysicalGroup(2, [1], name="My surface")
        gmsh.model.mesh.generate(2)
        gmsh.write(self.file_path)
        gmsh.finalize()
        self.num_boundary = len(self.lines)
        return fd.Mesh(self.file_path)

    def get_rand(self, mean, interval):
        return random.uniform(mean - interval, mean + interval)

    def get_rand_points(self):
        points = []
        split_p = np.random.uniform(0, 1, 4)
        # edge 1
        if (split_p[0] < self.split_threshold):
            points.append(
                [self.get_rand(self.quater, self.quater_interval), 0])
            points.append(
                [self.get_rand(self.three_quater, self.quater_interval), 0]
            )
        else:
            points.append([self.get_rand(self.mid, self.mid_interval), 0])
        # edge 2
        if (split_p[1] < self.split_threshold):
            points.append(
                [self.scale, self.get_rand(self.quater, self.quater_interval)]
            )
            points.append(
                [self.scale, self.get_rand(
                    self.three_quater, self.quater_interval)])
        else:
            points.append(
                [self.scale, self.get_rand(self.mid, self.mid_interval)]
            )
        # edge 3
        if (split_p[2] < self.split_threshold):
            points.append(
                [self.get_rand(self.three_quater, self.quater_interval),
                 self.scale]
            )
            points.append(
                [self.get_rand(self.quater, self.quater_interval),
                 self.scale]
            )
        else:
            points.append(
                [self.get_rand(self.mid, self.mid_interval),
                 self.scale]
            )
        # edge 4
        if (split_p[3] < self.split_threshold):
            points.append([
                0, self.get_rand(self.three_quater, self.quater_interval)])
            points.append([
                0, self.get_rand(self.quater, self.quater_interval)])
        else:
            points.append(
                [0, self.get_rand(self.mid, self.mid_interval)]
            )
            # points.append(p1)
        self.raw_points = points
        return

    def get_points(self):
        temp = []
        for i in range(len(self.raw_points)):
            temp.append(gmsh.model.geo.addPoint(
                self.raw_points[i][0], self.raw_points[i][1], 0, self.lc))
        self.points = temp

    def get_line(self):
        for i in range(len(self.points)):
            if (i < len(self.points) - 1):
                line = gmsh.model.geo.addLine(
                    self.points[i], self.points[i + 1])
                self.lines.append(line)
            else:
                line = gmsh.model.geo.addLine(self.points[i], self.points[0])
                self.lines.append(line)
        return

    def get_boundaries(self):
        print("in get_boundaries lines:", self.lines)
        for i, line_tag in enumerate(self.lines):
            gmsh.model.addPhysicalGroup(1, [line_tag], i + 1)
            gmsh.model.setPhysicalName(1, i + 1, "Boundary " + str(i + 1))

    def get_curve(self):
        gmsh.model.geo.addCurveLoop(
            [i for i in range(1, len(self.points) + 1)], 1)

    def get_plane(self):
        gmsh.model.geo.addPlaneSurface([1], 1)

    def show(self, file_path):
        mesh = fd.Mesh(file_path)
        fig = fd.triplot(mesh)
        return fig


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     mesh_gen = RandPolyMesh()
#     mesh_coarse = mesh_gen.get_mesh(res=5e-2, file_path="./temp1.msh")
#     mesh_fine = mesh_gen.get_mesh(res=4e-2, file_path="./temp2.msh")
#     mesh_gen.show('./temp1.msh')
#     mesh_gen.show('./temp2.msh')
#     plt.show()
