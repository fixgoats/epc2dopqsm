import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.common import npnormSqr
from src.penrose import filterByRadius, goldenRatio, makeSunGrid

path = os.path.join("graphs", "fig1.pdf")
a = np.load(os.path.join("data", "fig1repro", "psidata.npy"), allow_pickle=True).item()
psir = a["rpsidata"]
psir /= np.max(psir)
extentr = a["extentr"]
start = np.shape(psir)[0] // 6
end = np.shape(psir)[0] - start
extentr = [2 * x / 3 for x in extentr]
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(w=8, h=4)
# fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.9)
# cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.75))
# cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.75))
im0 = ax[0].imshow(
    npnormSqr(psir[start:end, start:end]),
    origin="lower",
    interpolation="none",
    extent=extentr,
)
fig.colorbar(im0)
points = filterByRadius(makeSunGrid(13.2 * goldenRatio**4, 4), 76)
mask = [np.abs(np.arctan2(y, x)) < np.pi / 4.5 for x, y in points]
points = points[mask]
ax[0].scatter(points[:, 0], points[:, 1], s=9, c="black", linewidths=0.0)
ax[0].scatter(points[:, 0], points[:, 1], s=6, c="orange", linewidths=0.0)

ax[0].set_title(r"$|\psi_r|^2$")
ax[0].set_xlabel(r"x [µm]")
ax[0].set_ylabel(r"y [µm]")

psik = a["kpsidata"]
psik /= np.max(psik)
extentk = a["extentk"]
startk = int(np.shape(psik)[0] / 2.5)
endk = np.shape(psik)[0] - startk
extentk = [x / 5 for x in extentk]
im1 = ax[1].imshow(
    npnormSqr(psik[startk:endk, startk:endk]),
    origin="lower",
    interpolation="none",
    extent=extentk,
    norm=LogNorm(vmin=np.exp(-10), vmax=1.0),
)
ax[1].set_xlabel(r"$k_x$ [µ$m^{-1}$]")
ax[1].set_ylabel(r"$k_y$ [µ$m^{-1}$]")
ax[1].set_title(r"$|\psi_k|^2$")
fig.suptitle("N=131")
fig.colorbar(im1)

fig.tight_layout()
fig.savefig(path)
