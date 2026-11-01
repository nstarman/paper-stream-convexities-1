"""Shared test configuration.

Double precision is enabled before any other import touches JAX: the curvature
tests compare against closed forms at tolerances float32 cannot meet, and the
figure scripts run in x64 too.
"""

import jax

jax.config.update("jax_enable_x64", True)
