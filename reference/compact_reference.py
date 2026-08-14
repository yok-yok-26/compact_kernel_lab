#!/usr/bin/env python3
import numpy as np


def compact_reference(values: np.ndarray, keep_flags: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    keep_flags = np.asarray(keep_flags, dtype=np.uint8)
    return values[keep_flags != 0].copy()
