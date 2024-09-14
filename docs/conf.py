# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os
import sys

sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, ".."))))
sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, "."))))
sys.path.insert(1, os.path.abspath(os.sep.join((os.curdir, "../warpmesh"))))
sys.path.insert(1, os.path.abspath(os.sep.join((os.curdir, "./warpmesh"))))


project = "warpmesh"
author = "Chunyang Wang"
release = "0.1"
latex_elements = {
    "preamble": "\\usepackage[utf8x]{inputenc}",
}
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
autodoc_mock_imports = ["torch", "firedrake", "numpy", "torch_geometric"]
autoclass_content = "both"
