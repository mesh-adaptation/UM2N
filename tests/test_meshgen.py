"""
Unit tests for the meshgen mesh generator module.
"""

from UM2N.generator.meshgen import (
    RandPolyMeshGenerator,
    UnstructuredUnitSquareMeshGenerator,
)
from firedrake.bcs import DirichletBC
import os
import pytest


@pytest.fixture(params=[1, 2, 3, 4])
def num_elem_bnd(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=["delaunay", "frontal"])
def mesh_algorithm(request):
    return request.param


@pytest.fixture(params=[RandPolyMeshGenerator, UnstructuredUnitSquareMeshGenerator])
def generator(request):
    return request.param


def test_boundary_segments(generator):
    """
    Check that the boundary segments are tagged with integers counting from 1.
    """
    file_path = "./tmp.msh"
    mesh_gen = generator(mesh_algorithm)
    mesh = mesh_gen.generate_mesh(res=1, file_path=file_path, remove_file=True)
    mesh.init()
    boundary_ids = mesh.exterior_facets.unique_markers
    assert (
        set(boundary_ids).difference({i + 1 for i in range(len(boundary_ids))}) == set()
    )
    assert not os.path.exists(file_path)


def test_num_points_boundaries_unitsquare(num_elem_bnd, mesh_algorithm):
    """
    Check that the numbers of points on each boundary segment of a unit square mesh are
    as expected.
    """
    file_path = "./tmp.msh"
    mesh_gen = UnstructuredUnitSquareMeshGenerator(mesh_type=mesh_algorithm)
    mesh = mesh_gen.generate_mesh(
        res=1 / num_elem_bnd, file_path=file_path, remove_file=True
    )
    mesh.init()
    boundary_ids = mesh.exterior_facets.unique_markers
    for boundary_id in boundary_ids:
        dbc = DirichletBC(mesh.coordinates.function_space(), 0, boundary_id)
        assert len(dbc.nodes) == num_elem_bnd + 1
    assert not os.path.exists(file_path)
