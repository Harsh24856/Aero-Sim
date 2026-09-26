"""One call to seed Python, NumPy and TensorFlow, returning the seed for the run JSON."""
from __future__ import annotations

import secrets


def set_random_seed(seed: int | None = None) -> int:
    """Seed everything keras.utils.set_random_seed covers; draw a seed if none is given.
    Record the returned value in the run's JSON so the run can be repeated."""
    import keras
    seed = int(secrets.randbelow(2**31 - 1)) if seed is None else int(seed)
    keras.utils.set_random_seed(seed)
    return seed
