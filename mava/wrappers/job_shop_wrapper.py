from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment

from jumanji.environments.packing.job_shop import JobShop

env = JobShop()
print(type(env))