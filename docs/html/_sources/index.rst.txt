.. # Author: Chunyang Wang
.. # GitHub Username: acse-cw1722

WarpMesh: A Machin Learning Based Mesh Movement Package
********************************************************

WarpMesh package docs
========================================================

============
.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: generator
  :members: MeshGenerator, HelmholtzSolver, RandomHelmholtzGenerator

.. automodule:: processor
  :members: MeshProcessor

.. automodule:: model
  :members:  MRN, GlobalFeatExtractor, LocalFeatExtractor, RecurrentGATConv, train, evaluate, load_model, TangleCounter

.. automodule:: loader
  :members: MeshDataset, MeshData, normalise