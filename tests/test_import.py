# Author: Chunyang Wang
# GitHub Username: acse-cw1722

from pytest import fixture


@fixture(scope="module")
def UM2N():
    import UM2N

    return UM2N


@fixture(scope="module")
def firedrake():
    import firedrake

    return firedrake


@fixture(scope="module")
def movement():
    import movement

    return movement


def test_import(UM2N, firedrake, movement):
    assert UM2N
    assert firedrake
    assert movement
