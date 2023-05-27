from pathlib import Path
import dolfin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py

from geometry import load_geometry


class CylinderSlice(dolfin.UserExpression):
    def __init__(self, f):
        self.f = f
        super().__init__()

    def eval(self, value, x):
        dx = 0.05
        values = [self.f(0, x[0], x[1])]
        if not np.isclose(x[0], 1, atol=dx):
            values.append(self.f(0, x[0] + dx, x[1]))
        if not np.isclose(x[1], 1, atol=dx):
            values.append(self.f(0, x[0], x[1] + dx))
        if not np.isclose(x[0], -1, atol=dx):
            values.append(self.f(0, x[0] - dx, x[1]))
        if not np.isclose(x[1], -1, atol=dx):
            values.append(self.f(0, x[0], x[1] - dx))
        value[0] = np.median(values)

    def value_shape(self):
        return ()


def postprocess_basic(resultsdir="results", datadir="data", figdir="figures"):
    output = Path(resultsdir) / "results.xdmf"
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")

    disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
    V_disk = dolfin.FunctionSpace(disk, "CG", 2)

    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 2)
    V_CG1 = dolfin.FunctionSpace(geo.mesh, "CG", 1)

    sigma_xx = dolfin.Function(V_DG1)
    sigma_r = dolfin.Function(V_DG1)
    sigma_c = dolfin.Function(V_DG1)
    E_xx = dolfin.Function(V_DG1)
    E_r = dolfin.Function(V_DG1)
    E_c = dolfin.Function(V_DG1)
    p = dolfin.Function(V_CG1)

    sigma_xx.set_allow_extrapolation(True)
    sigma_r.set_allow_extrapolation(True)
    sigma_c.set_allow_extrapolation(True)
    E_xx.set_allow_extrapolation(True)
    E_r.set_allow_extrapolation(True)
    E_c.set_allow_extrapolation(True)
    p.set_allow_extrapolation(True)

    t = np.load(Path(resultsdir) / "t.npy")
    gamma = np.load(Path(resultsdir) / "gamma.npy")

    with h5py.File(output.with_suffix(".h5"), "r") as f:
        N = len(f["u"].keys())

    points = np.arange(0, geo.mesh.coordinates().max(0)[2] + 0.1, 0.1)

    sigma_xx_arr = np.zeros((N, len(points)))
    sigma_r_arr = np.zeros((N, len(points)))
    sigma_c_arr = np.zeros((N, len(points)))
    E_xx_arr = np.zeros((N, len(points)))
    E_r_arr = np.zeros((N, len(points)))
    E_c_arr = np.zeros((N, len(points)))
    p_arr = np.zeros((N, len(points)))

    Path(figdir).mkdir(exist_ok=True, parents=True)

    with dolfin.XDMFFile(output.as_posix()) as f:
        for i in range(N):
            f.read_checkpoint(sigma_xx, "sigma_xx", i)
            f.read_checkpoint(sigma_r, "sigma_r", i)
            f.read_checkpoint(sigma_c, "sigma_c", i)
            f.read_checkpoint(E_xx, "E_xx", i)
            f.read_checkpoint(E_r, "E_r", i)
            f.read_checkpoint(E_c, "E_c", i)
            f.read_checkpoint(p, "p", i)

            for j, point in enumerate(points):
                sigma_xx_arr[i, j] = sigma_xx(0, 0, point)
                sigma_r_arr[i, j] = sigma_r(0, 0, point)
                sigma_c_arr[i, j] = sigma_c(0, 0, point)
                E_xx_arr[i, j] = E_xx(0, 0, point)
                E_r_arr[i, j] = E_r(0, 0, point)
                E_c_arr[i, j] = E_c(0, 0, point)
                p_arr[i, j] = p(0, 0, point)

            if i == gamma.argmax():
                for func, title, name in [
                    (sigma_xx, r"$\sigma_{xx}$", "sigma_xx"),
                    (sigma_r, r"$\sigma_{r}$", "sigma_r"),
                    (sigma_c, r"$\sigma_{c}$", "sigma_c"),
                    (E_xx, r"$E_{xx}$", "E_xx"),
                    (E_r, r"$E_{r}$", "E_r"),
                    (E_c, r"$E_{c}$", "E_c"),
                    (p, r"$p$", "p"),
                ]:
                    plt.colorbar(
                        dolfin.plot(dolfin.interpolate(CylinderSlice(func), V_disk))
                    )
                    plt.title(title)
                    plt.savefig(Path(figdir) / f"{name}.png")
                    plt.close()

    fig, ax = plt.subplots(3, 3, sharex=True, figsize=(10, 8))

    lines = []
    labels = []
    for j, point in enumerate(points):
        ax[0, 0].plot(t, sigma_xx_arr[:, j], color=cm.tab20(point / max(points)))
        ax[1, 0].plot(t, sigma_r_arr[:, j], color=cm.tab20(point / max(points)))
        ax[2, 0].plot(t, sigma_c_arr[:, j], color=cm.tab20(point / max(points)))

        ax[0, 1].plot(t, E_xx_arr[:, j], color=cm.tab20(point / max(points)))
        ax[1, 1].plot(t, E_r_arr[:, j], color=cm.tab20(point / max(points)))
        ax[2, 1].plot(t, E_c_arr[:, j], color=cm.tab20(point / max(points)))

        (l,) = ax[0, 2].plot(t, p_arr[:, j], color=cm.tab20(point / max(points)))
        lines.append(l)
        labels.append(f"$r = {point:.1f}$")

    ax[0, 0].set_title(r"$\sigma_{xx}$")
    ax[1, 0].set_title(r"$\sigma_{r}$")
    ax[2, 0].set_title(r"$\sigma_{c}$")

    ax[0, 1].set_title(r"$E_{xx}$")
    ax[1, 1].set_title(r"$E_{r}$")
    ax[2, 1].set_title(r"$E_{c}$")

    ax[0, 2].set_title("$p$")
    ax[1, 2].plot(t, gamma)
    ax[1, 2].set_title(r"$\gamma$")

    fig.subplots_adjust(right=0.87)
    fig.legend(lines, labels, loc="center right")
    fig.savefig(Path(figdir) / "results.png")


def postprocess_effect_of_spring(
    resultdirs, datadir="data", figdir="figures", plot_slice=True
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")

    disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
    V_disk = dolfin.FunctionSpace(disk, "CG", 2)

    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 2)
    V_CG1 = dolfin.FunctionSpace(geo.mesh, "CG", 1)

    sigma_xx = dolfin.Function(V_DG1)
    sigma_r = dolfin.Function(V_DG1)
    sigma_c = dolfin.Function(V_DG1)
    E_xx = dolfin.Function(V_DG1)
    E_r = dolfin.Function(V_DG1)
    E_c = dolfin.Function(V_DG1)
    p = dolfin.Function(V_CG1)

    sigma_xx.set_allow_extrapolation(True)
    sigma_r.set_allow_extrapolation(True)
    sigma_c.set_allow_extrapolation(True)
    E_xx.set_allow_extrapolation(True)
    E_r.set_allow_extrapolation(True)
    E_c.set_allow_extrapolation(True)
    p.set_allow_extrapolation(True)

    num_dirs = len(resultdirs)
    fst = next(iter(resultdirs.values()))
    t = np.load(Path(fst) / "t.npy")
    gamma = np.load(Path(fst) / "gamma.npy")
    N = len(t)

    points = np.arange(0, 1.1, 0.1)
    sigma_xx_arr = np.zeros((num_dirs, N, len(points)))
    sigma_r_arr = np.zeros((num_dirs, N, len(points)))
    sigma_c_arr = np.zeros((num_dirs, N, len(points)))
    E_xx_arr = np.zeros((num_dirs, N, len(points)))
    E_r_arr = np.zeros((num_dirs, N, len(points)))
    E_c_arr = np.zeros((num_dirs, N, len(points)))
    p_arr = np.zeros((num_dirs, N, len(points)))

    for idx, (spring, resultdir) in enumerate(resultdirs.items()):
        output = Path(resultdir) / "results.xdmf"

        with dolfin.XDMFFile(output.as_posix()) as f:
            for i in range(N):
                f.read_checkpoint(sigma_xx, "sigma_xx", i)
                f.read_checkpoint(sigma_r, "sigma_r", i)
                f.read_checkpoint(sigma_c, "sigma_c", i)
                f.read_checkpoint(E_xx, "E_xx", i)
                f.read_checkpoint(E_r, "E_r", i)
                f.read_checkpoint(E_c, "E_c", i)
                f.read_checkpoint(p, "p", i)

                for j, point in enumerate(points):
                    sigma_xx_arr[idx, i, j] = sigma_xx(0, 0, point)
                    sigma_r_arr[idx, i, j] = sigma_r(0, 0, point)
                    sigma_c_arr[idx, i, j] = sigma_c(0, 0, point)
                    E_xx_arr[idx, i, j] = E_xx(0, 0, point)
                    E_r_arr[idx, i, j] = E_r(0, 0, point)
                    E_c_arr[idx, i, j] = E_c(0, 0, point)
                    p_arr[idx, i, j] = p(0, 0, point)

                if i == gamma.argmax() and plot_slice:
                    for func, title, name in [
                        (sigma_xx, rf"$\sigma_{{xx}}$ (spring={spring})", "sigma_xx"),
                        (sigma_r, rf"$\sigma_{{r}}$ (spring={spring})", "sigma_r"),
                        (sigma_c, rf"$\sigma_{{c}}$ (spring={spring})", "sigma_c"),
                        (E_xx, rf"$E_{{xx}}$ (spring={spring})", "E_xx"),
                        (E_r, rf"$E_{{r}}$ (spring={spring})", "E_r"),
                        (E_c, rf"$E_{{c}}$ (spring={spring})", "E_c"),
                        (p, rf"$p$ (spring={spring})", "p"),
                    ]:
                        plt.colorbar(
                            dolfin.plot(dolfin.interpolate(CylinderSlice(func), V_disk))
                        )
                        plt.title(title)
                        plt.savefig(f"figures/{name}_{spring}.png")
                        plt.close()

    fig_sigma, ax_sigma = plt.subplots(
        3, num_dirs, sharex=True, sharey="row", figsize=(10, 8)
    )
    fig_E, ax_E = plt.subplots(3, num_dirs, sharex=True, sharey="row", figsize=(10, 8))
    fig_p, ax_p = plt.subplots(1, num_dirs, sharex=True, sharey="row", figsize=(10, 8))

    for idx, spring in enumerate(resultdirs.keys()):
        ax_sigma[0, idx].set_title(f"spring = {spring}")
        ax_E[0, idx].set_title(f"spring = {spring}")
        ax_p[idx].set_title(f"spring = {spring}")
        lines = []
        labels = []
        for j, point in enumerate(points):
            ax_sigma[0, idx].plot(t, sigma_xx_arr[idx, :, j], color=cm.tab20(point))
            ax_sigma[1, idx].plot(t, sigma_r_arr[idx, :, j], color=cm.tab20(point))
            ax_sigma[2, idx].plot(t, sigma_c_arr[idx, :, j], color=cm.tab20(point))

            ax_E[0, idx].plot(t, E_xx_arr[idx, :, j], color=cm.tab20(point))
            ax_E[1, idx].plot(t, E_r_arr[idx, :, j], color=cm.tab20(point))
            ax_E[2, idx].plot(t, E_c_arr[idx, :, j], color=cm.tab20(point))

            (l,) = ax_p[idx].plot(t, p_arr[idx, :, j], color=cm.tab20(point))
            lines.append(l)
            labels.append(f"$r = {point:.1f}$")

    for i, label in enumerate([r"$\sigma_{xx}$", r"$\sigma_{r}$", r"$\sigma_{c}$"]):
        ax_sigma2 = ax_sigma[i, idx].twinx()
        ax_sigma2.set_ylabel(label)
        ax_sigma2.set_yticks([])

    for i, label in enumerate([r"$E_{xx}$", r"$E_{r}$", r"$E_{c}$"]):
        ax_E2 = ax_E[i, idx].twinx()
        ax_E2.set_ylabel(label)
        ax_E2.set_yticks([])

    ax_p2 = ax_p[idx].twinx()
    ax_p2.set_ylabel("$p$")
    ax_p2.set_yticks([])

    for fig in [fig_p, fig_E, fig_sigma]:
        fig.subplots_adjust(right=0.87)
        fig.legend(lines, labels, loc="center right")
    fig_sigma.savefig(Path(figdir) / "results_spring_sigma.png")
    fig_E.savefig(Path(figdir) / "results_spring_E.png")
    fig_p.savefig(Path(figdir) / "results_spring_p.png")
