"""Tools to visualize all-sky Compton telescope data"""

__author__ = "Henricke Fleischhack, Eric Charles"
__author_email__ = "badass@stanford.edu"
__url__ = "https://github.com/eacharles/comsky"
__desc__ = "Tools to visualize all-sky Compton telescope data"

import os

try:
    from .version import get_git_version
    __version__ = get_git_version()
except Exception as message: #pragma: no cover
    print(message)


from . import utils
