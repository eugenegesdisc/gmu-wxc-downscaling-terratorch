
"""
    The module is to search for modules under "gmudowncalingterratorch.subcommands"
    starting with "gmu_downscaling_subcommand_".
"""

import importlib
import pkgutil

from .. import subcommands

def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__+".")


discovered_subcommand_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(subcommands)
    if name.startswith("gmudownscalingterratorch.subcommands."+"gmu_downscaling_subcommand_")
}
