import numpy as np


def random_road(rng, amplitude=0.01):
    return float(rng.normal(0.0, amplitude))


def sinusoidal_road(t, amplitude=0.01, frequency=1.0):
    return float(amplitude * np.sin(2.0 * np.pi * frequency * t))


def bump_road(t, amplitude=0.02, start=1.0, end=1.2):
    if start <= t <= end:
        return float(amplitude)
    return 0.0


def pothole_road(t, amplitude=0.02, start=1.0, end=1.2):
    if start <= t <= end:
        return float(-amplitude)
    return 0.0