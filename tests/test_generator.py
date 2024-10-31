from UM2N.generator.squaremesh import UnstructuredUnitSquareMesh
from firedrake.bcs import DirichletBC
import os
import pytest


@pytest.fixture(params=[1, 2, 3, 4])
def num_elem_bnd(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=["delaunay", "frontal"])
def mesh_algorithm(request):
    return request.param


def test_boundary(num_elem_bnd, mesh_algorithm):
    file_path = "./tmp.msh"
    mesh_gen = UnstructuredUnitSquareMesh(mesh_type=mesh_algorithm)
    mesh = mesh_gen.generate_mesh(
        res=1 / num_elem_bnd, file_path=file_path, remove_file=True
    )
    mesh.init()
    boundary_ids = mesh.exterior_facets.unique_markers
    assert set(boundary_ids).difference({1, 2, 3, 4}) == set()
    for boundary_id in boundary_ids:
        dbc = DirichletBC(mesh.coordinates.function_space(), 0, boundary_id)
        assert len(dbc.nodes) == num_elem_bnd + 1
    assert not os.path.exists(file_path)
