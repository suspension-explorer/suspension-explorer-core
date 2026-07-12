"""Public YAML loading adapter."""

from kinematics.cli.io.loaders import load_geometry as load_geometry
from kinematics.cli.io.sweep_loader import parse_sweep_file

load_sweep = parse_sweep_file
