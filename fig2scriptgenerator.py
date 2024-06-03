import numpy as np

from src.penrose import goldenRatio

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 10.4  # 16 is ca. condensation threshold
dt = 0.05
Gamma = 0.1
eta = 2
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32
N = 1024
startX = -120
endX = 120
dx = (endX - startX) / N
prerun = 12000
D = 13.2
radius = D * goldenRatio**4
kmax = np.pi / dx
dk = 2 * kmax / N
sigmax = 1.27
sigmay = 1.27
seed = 20011204

a = f"""
import math
import json
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
params = {{
    "gammalp": {gammalp},
    "alpha": {alpha},
    "G": {G},
    "R": {R},
    "Gamma": {Gamma},
    "eta": {eta},
    "D": {radius},
    "m": {m},
    "N": {N},
    "pumpStrength": {pumpStrength},
    "startX": {startX},
    "endX": {endX},
    "prerun": {prerun},
    "sigmax": {sigmax},
    "sigmay": {sigmay},
}}


@torch.jit.script
def tnormSqr(x):
    return x.conj() * x


@torch.jit.script
def V(psi, nR, constPart):
    return constPart + {alpha} * tnormSqr(psi) + ({G} + 0.5j * {R}) * nR


@torch.jit.script
def halfRTimeEvo(psi, nR, constPart):
    return torch.exp(-0.5j * {dt} * V(psi, nR, constPart))


@torch.jit.script
def stepNR(psi, nR, pump):
    return (
        math.exp(-{dt} * {Gamma}) * torch.exp((-{dt} * {R}) * tnormSqr(psi)) * nR
        + pump * {dt}
    )


@torch.jit.script
def step(psi0, nR0, kTimeEvo, constPart, pump):
    rProp = halfRTimeEvo(psi0, nR0, constPart)
    psi = rProp * psi0
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi))
    psi = rProp * psi
    nR = stepNR(psi, nR0, pump)
    return psi, nR


@torch.jit.script
def runSim(psi, nR, kTimeEvo, constPart, pump, npolars):
    for i in range({prerun}):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
        npolars[i] = torch.sum(tnormSqr(psi).real)
    return psi, nR


nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
k = torch.arange({-kmax}, {kmax}, {dk}, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * {hbar * dt / m} * (kxv * kxv + kyv * kyv))
basedir = os.path.join("data", "fig2repro")
Path(basedir).mkdir(parents=True, exist_ok=True)
with open(os.path.join(basedir, "parameters.json"), "w") as f:
    json.dump(params, f)
x = np.arange({startX}, {endX}, {dx})
xv, yv = np.meshgrid(x, x)
xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
cutoffs = {{"pn46": 46, "pn86": 60, "pn111": 70, "pn151": 80}}
for key, value in cutoffs.items():
    rng = np.random.default_rng({seed})
    psi = torch.from_numpy(smoothnoise(xv, yv, rng)).type(dtype=torch.cfloat).to(device='cuda')
    nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
    pump = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
    
    points = filterByRadius(makeSunGrid({radius}, 4), value)
    for p in points:
        pump += {pumpStrength} * tgauss(xv - p[0],
                                        yv - p[1],
                                        sigmax={sigmax},
                                        sigmay={sigmay})
    
    constpart = {constV} + {G * eta / Gamma} * pump
    npolarsgpu = torch.zeros(({prerun}), dtype=torch.float, device="cuda")
    psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu)
    npolars = npolarsgpu.detach().cpu().numpy()
    np.save(os.path.join(basedir, f"npolars{{key}}"), npolars)
    kpsidata = tfft.fftshift(tfft.fft2(psi)).detach().cpu().numpy()
    rpsidata = psi.detach().cpu().numpy()
    extentr = np.array([{startX}, {endX}, {startX}, {endX}])
    extentk = np.array([{-kmax}, {kmax}, {-kmax}, {kmax}])
    np.save(os.path.join(basedir, f"psidata{{key}}"),
            {{"kpsidata": kpsidata,
             "rpsidata": rpsidata,
             "extentr": extentr,
             "extentk": extentk,
             }})


t2 = time.time()
print(f"finished in {{t2 - t1}} seconds")
"""

with open("fig2script.py", "w") as f:
    f.write(a)
