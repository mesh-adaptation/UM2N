# Author: Chunyang Wang
# GitHub Username: acse-cw1722

from pytest import fixture


@fixture(scope="module")
def warpmesh():
    import warpmesh

    return warpmesh


@fixture(scope="module")
def firedrake():
    import firedrake

    return firedrake


@fixture(scope="module")
def movement():
    import movement

    return movement


def test_import(warpmesh, firedrake, movement):
    assert warpmesh
    assert firedrake
    assert movement
