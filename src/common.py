# Classes to time evolve initial condition according to the Schrödinger equation

import numpy as np
import torch
from numba import complex128, float64, vectorize
from scipy.signal import convolve2d

hbar = 6.582119569e-1  # meV * ps


# Using numba for this is actually faster than the default, but vectorizing
# with numpy is somehow slower than default. This could be made even faster by using njit
# with known array dimensions, but then you'd need distinct functions for vectors
# and matrices.
@vectorize([float64(complex128)])
def npnormSqr(x):
    return x.real * x.real + x.imag * x.imag


# For use with torch. This case doesn't benefit from jit and such and
# the conj version is just as fast as the expanded version
def tnormSqr(x):
    return x.conj() * x


# Create a somewhat more realistic initial condition than white noise
def smoothnoise(xv, yv):
    rng = np.random.default_rng()
    random = rng.uniform(-1, 1, np.shape(xv)) + 1j * rng.uniform(-1, 1, np.shape(xv))
    krange = np.linspace(-2, 2, num=21)
    kbasex, kbasey = np.meshgrid(krange, krange)
    kernel = gauss(kbasex, kbasey)
    kernel /= np.sum(kernel)
    output = convolve2d(random, kernel, mode="same")
    output = convolve2d(output, kernel, mode="same")
    return output / np.sqrt(np.sum(npnormSqr(output)))


def gauss(x, y, sigmax=1.0, sigmay=1.0):
    return np.exp(-((x / sigmax) ** 2) - (y / sigmay) ** 2)


def tgauss(x, y, sigmax=1.0, sigmay=1.0):
    return torch.exp(-((x / sigmax) ** 2) - (y / sigmay) ** 2)
