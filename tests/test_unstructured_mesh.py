"""
Unit tests for the generate_mesh mesh generator module.
"""

from UM2N.generator.unstructured_mesh import (
    UnstructuredRandomPolygonalMeshGenerator,
    UnstructuredSquareMeshGenerator,
)
from firedrake.assemble import assemble
from firedrake.bcs import DirichletBC
from firedrake.constant import Constant
import numpy as np
import os
import pytest
import ufl


@pytest.fixture(params=[1, 2, 3, 4])
def num_elem_bnd(request):
    return request.param


@pytest.fixture(params=[1, 10, 0.2, np.pi])
def scale(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=["delaunay", "frontal"])
def mesh_algorithm(request):
    return request.param


@pytest.fixture(
    params=[
        UnstructuredRandomPolygonalMeshGenerator,
        UnstructuredSquareMeshGenerator,
    ]
)
def generator(request):
    return request.param


def generate_mesh(generator, mesh_algorithm, scale=1.0, **kwargs):
    """
    Utility mesh generator function for testing purposes.
    """
    mesh_gen = generator(mesh_type=mesh_algorithm, scale=scale)
    kwargs.setdefault("remove_file", True)
    mesh = mesh_gen.generate_mesh(**kwargs)
    mesh.init()
    return mesh


def test_file_removal():
    """
    Test that the remove_file keyword argument works as expected.
    """
    output_filename = "./tmp.msh"
    assert not os.path.exists(output_filename)
    generate_mesh(
        UnstructuredSquareMeshGenerator,
        1,
        res=1.0,
        output_filename=output_filename,
        remove_file=False,
    )
    assert os.path.exists(output_filename)
    os.remove(output_filename)
    assert not os.path.exists(output_filename)
    generate_mesh(
        UnstructuredSquareMeshGenerator, 1, res=1.0, output_filename=output_filename
    )
    assert not os.path.exists(output_filename)


def test_boundary_segments(generator):
    """
    Check that the boundary segments are tagged with integers counting from 1.
    """
    mesh = generate_mesh(generator, 1, res=1.0)
    boundary_ids = mesh.exterior_facets.unique_markers
    assert (
        set(boundary_ids).difference({i + 1 for i in range(len(boundary_ids))}) == set()
    )


def test_num_points_boundaries_square(num_elem_bnd, mesh_algorithm):
    """
    Check that the numbers of points on each boundary segment of a unit square mesh are
    as expected.
    """
    mesh = generate_mesh(UnstructuredSquareMeshGenerator, 1, res=1.0 / num_elem_bnd)
    boundary_ids = mesh.exterior_facets.unique_markers
    for boundary_id in boundary_ids:
        dbc = DirichletBC(mesh.coordinates.function_space(), 0, boundary_id)
        assert len(dbc.nodes) == num_elem_bnd + 1


def test_area_squaremesh(num_elem_bnd, mesh_algorithm, scale):
    """
    Check that the area of a square mesh is equal to the scale factor squared.
    """
    mesh = generate_mesh(
        UnstructuredSquareMeshGenerator, 1, res=1.0 / num_elem_bnd, scale=scale
    )
    assert np.isclose(assemble(Constant(1.0, domain=mesh) * ufl.dx), scale**2)


def test_num_cells_with_res_and_scale(generator, num_elem_bnd, mesh_algorithm):
    """
    Check that doubling or halving the overall resolution doesn't affect the number of
    cells for the square mesh, so long as the resolution is changed accordingly.
    """
    generator = UnstructuredSquareMeshGenerator
    mesh1 = generate_mesh(generator, mesh_algorithm, res=1.0 / num_elem_bnd)
    mesh2 = generate_mesh(generator, mesh_algorithm, res=2.0 / num_elem_bnd, scale=2.0)
    meshp5 = generate_mesh(generator, mesh_algorithm, res=0.5 / num_elem_bnd, scale=0.5)
    assert np.allclose((mesh2.num_cells(), meshp5.num_cells()), mesh1.num_cells())
