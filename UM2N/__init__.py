import os

os.environ["OMP_NUM_THREADS"] = "1"

from pkg_resources import DistributionNotFound, get_distribution  # noqa

from .processor import *  # noqa
from .generator import *  # noqa
from .model import *  # noqa
from .loader import *  # noqa
from .helper import *  # noqa
from .test import *  # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass