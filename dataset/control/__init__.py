# Importing dataset.control (or any of its submodules, since Python runs this file
# first) must register every control signal, regardless of which one a caller asks for
# by name -- otherwise control_registry.get("ddsp_controls") fails unless something
# already happened to import ddsp_guitar.py directly first. Mirrors
# dataset/latent/__init__.py, which exists for exactly this reason.
#
# signal first: ddsp_guitar imports ControlSignal/control_registry from it.
from dataset.control import signal, ddsp_guitar  # noqa: F401
