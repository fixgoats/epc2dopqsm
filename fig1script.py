import json
import math
import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.common import smoothnoise, tgauss
from src.penrose import filterByRadius, makeSunGrid

t1 = time.time()
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
params = {
    "gammalp": 0.2,
    "alpha": 0.0004,
    "G": 0.002,
    "R": 0.016,
    "Gamma": 0.1,
    "eta": 2,
    "D": 90.47414595449584,
    "cutoff": 76,
    "m": 0.32,
    "N": 1024,
    "pumpStrength": 22.4,
    "startX": -120,
    "endX": 120,
    "nsteps": 8000,
    "sigmax": 1.27,
    "sigmay": 1.27,
}


@torch.jit.script
def tnormSqr(x):
    return x.conj() * x


@torch.jit.script
def V(psi, nR, constPart):
    return constPart + 0.0004 * tnormSqr(psi) + (0.002 + 0.5j * 0.016) * nR


@torch.jit.script
def halfRStepPsi(psi, nR, constPart):
    return psi * torch.exp(-0.5j * 0.05 * V(psi, nR, constPart))


@torch.jit.script
def halfStepNR(psi, nR, pump):
    return (
        math.exp(-0.5 * 0.05 * 0.1)
        * torch.exp((-0.5 * 0.05 * 0.016) * tnormSqr(psi))
        * nR
        + pump * 0.05 * 0.5
    )


@torch.jit.script
def stepNR(psi, nR, pump):
    return (
        math.exp(-0.05 * 0.1) * torch.exp((-0.05 * 0.016) * tnormSqr(psi)) * nR
        + pump * 0.05
    )


@torch.jit.script
def step(psi0, nR0, kTimeEvo, constPart, pump):
    psi = halfRStepPsi(psi0, nR0, constPart)
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi0))
    nR = halfStepNR(psi, nR0, pump)
    psi = halfRStepPsi(psi, nR, constPart)
    nR = halfStepNR(psi, nR, pump)
    return psi, nR


@torch.jit.script
def altstep(psi0, nR0, kTimeEvo, constPart, pump):
    psi = halfRStepPsi(psi0, nR0, constPart)
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi0))
    psi = halfRStepPsi(psi, nR0, constPart)
    nR = stepNR(psi, nR0, pump)
    return psi, nR


@torch.jit.script
def runSim(psi, nR, kTimeEvo, constPart, pump, npolars):
    for i in range(8000):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
        npolars[i] = torch.sum(tnormSqr(psi).real)
    return psi, nR


nR = torch.zeros((1024, 1024), device="cuda", dtype=torch.cfloat)
k = torch.arange(
    -13.40412865531645, 13.40412865531645, 0.02617993877991494, device="cuda"
).type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * 0.102845618265625 * (kxv * kxv + kyv * kyv))
basedir = os.path.join("data", "fig1repro")
Path(basedir).mkdir(parents=True, exist_ok=True)
x = np.arange(-120, 120, 0.234375)
xv, yv = np.meshgrid(x, x)
xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device="cuda")
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device="cuda")
rng = np.random.default_rng(20011204)
psi = (
    torch.from_numpy(smoothnoise(xv, yv, rng))
    .type(dtype=torch.cfloat)
    .to(device="cuda")
)

with open(os.path.join(basedir, "parameters.json"), "w") as f:
    json.dump(params, f)
nR = torch.zeros((1024, 1024), device="cuda", dtype=torch.cfloat)
pump = torch.zeros((1024, 1024), device="cuda", dtype=torch.cfloat)
points = filterByRadius(makeSunGrid(90.47414595449584, 4), 76)
print(np.shape(points))  # verify that the right number of points are used
for p in points:
    pump += 22.4 * tgauss(xv - p[0], yv - p[1], sigmax=1.27, sigmay=1.27)

constpart = -0.1j + 0.04 * pump
npolarsgpu = torch.zeros((8000), dtype=torch.float, device="cuda")
psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu)
npolars = npolarsgpu.detach().cpu().numpy()
np.save(os.path.join(basedir, "npolars"), npolars)
kpsidata = tfft.fftshift(tfft.fft2(psi)).detach().cpu().numpy()
rpsidata = psi.detach().cpu().numpy()
extentr = np.array([-120, 120, -120, 120])
extentk = np.array(
    [-13.40412865531645, 13.40412865531645, -13.40412865531645, 13.40412865531645]
)
np.save(
    os.path.join(basedir, "psidata"),
    {
        "kpsidata": kpsidata,
        "rpsidata": rpsidata,
        "extentr": extentr,
        "extentk": extentk,
    },
)


t2 = time.time()
print(f"finished in {t2 - t1} seconds")
