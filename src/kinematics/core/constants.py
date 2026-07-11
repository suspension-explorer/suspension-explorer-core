# Near-zero guard for numerical routines (e.g. division-by-zero protection when
# extracting an axis from a rotation vector, or checking if an angle is effectively
# zero). Must be tight because the solver evaluates at perturbations well below any
# geometric tolerance.
EPS_NUMERICAL = 1e-15

# Geometric tolerance for equality checks, zero-length vectors, parallelism, etc.
# Appropriate for mm-scale coordinates.
EPS_GEOMETRIC = 1e-6

# Minimum reliable signed volume for an authored handedness constraint.
MIN_CHIRALITY_VOLUME = 1e-6

# Solve tolerances.
SOLVE_TOLERANCE_VALUE = 1e-5  # 0.01um for mm units.
SOLVE_TOLERANCE_STEP = 1e-9
SOLVE_TOLERANCE_GRAD = 1e-9

# Maximum accepted absolute residual after optimizer convergence.
SOLVE_ACCEPT_RESIDUAL = 1e-3

# Tolerance for tests; has headroom over solve tolerances.
TEST_TOLERANCE = 1e-3

# Because rims are still spec'd in freedom units.
MM_PER_INCH = 25.4
