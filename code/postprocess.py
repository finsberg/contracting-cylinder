from pathlib import Path
import dolfin


import ufl_legacy as ufl
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
    V_disk = dolfin.FunctionSpace(disk, "CG", 1)

    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 1)
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


def name2latex(name: str) -> str:
    name_lst = name.split("_")
    if len(name_lst) == 1:
        return f"${name}$"

    *sym, sup = name_lst
    if len(sym) == 1:
        sym = sym[0]
        if sym == "sigma":
            sym = "\\sigma"
        return f"${sym}_{{{sup}}}$"
    else:
        if sym[0] == "sigma":
            sym = "\\mathrm{dev} \\sigma"
        else:
            raise ValueError(sym)
        return f"${sym}_{{{sup}}}$"


def postprocess_effect_of_compressibility_stress(
    resultdirs, datadir="data_basic", figdir="figures_comp", plot_slice=True
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True)
    disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
    R = geo.mesh.coordinates().max(0)[-1]
    disk.coordinates()[:] *= R
    V_disk = dolfin.FunctionSpace(disk, "CG", 2)

    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    V_CG1 = dolfin.FunctionSpace(geo.mesh, "CG", 1)

    funcs = {
        name: dolfin.Function(V_DG1)
        for name in [
            "sigma_xx",
            "sigma_r",
            "sigma_c",
            "sigma_dev_xx",
            "sigma_dev_r",
            "sigma_dev_c",
            "E_xx",
            "E_r",
            "E_c",
        ]
    }
    funcs["p"] = dolfin.Function(V_CG1)

    for v in funcs.values():
        v.set_allow_extrapolation(True)

    num_dirs = len(resultdirs)
    fst = next(iter(resultdirs.values()))
    t = np.load(Path(fst) / "t.npy")
    gamma = np.load(Path(fst) / "gamma.npy")
    N = len(t)

    points = np.linspace(0, R, 11)
    arrs = {name: np.zeros((num_dirs, N, len(points))) for name in funcs}
    shortening = np.zeros((num_dirs, N))
    meshvol = dolfin.assemble(dolfin.Constant(1) * dolfin.dx(geo.mesh))

    for idx, (kappa, resultdir) in enumerate(resultdirs.items()):
        output = Path(resultdir) / "results.xdmf"
        print(output)

        with dolfin.XDMFFile(output.as_posix()) as f:
            for i in range(N):
                for name, func in funcs.items():
                    print(name)
                    if name == "p" and kappa is not None:
                        continue

                    f.read_checkpoint(func, name, i)

                    if name == "E_xx":
                        shortening[idx, i] = (
                            dolfin.assemble(funcs["E_xx"] * dolfin.dx) / meshvol
                        )

                for j, point in enumerate(points):
                    for name, func in funcs.items():
                        arrs[name][idx, i, j] = func(0, 0, point)

                if i == gamma.argmax() and plot_slice:
                    for name, func in funcs.items():
                        if name == "p" and kappa is not None:
                            continue
                        extra = (
                            " (incomp)"
                            if kappa is None
                            else f" ($\\kappa={kappa:.0f}$)"
                        )
                        title = name2latex(name) + extra

                        plt.colorbar(
                            dolfin.plot(dolfin.interpolate(CylinderSlice(func), V_disk))
                        )
                        plt.title(title)
                        plt.savefig(figdir / f"{name}_{kappa}.png")
                        plt.close()

    fig, ax = plt.subplots()
    for idx, kappa in enumerate(resultdirs.keys()):
        title = "incomp" if kappa is None else rf"$\kappa = {kappa:.0f}$"
        ax.plot(t, shortening[idx, :], label=title)
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("shortening")
    fig.savefig(figdir / "results_comp_shortening.png")

    fig_sigma, ax_sigma = plt.subplots(
        3, num_dirs, sharex=True, sharey="row", figsize=(10, 8)
    )
    fig_sigma_dev, ax_sigma_dev = plt.subplots(
        3, num_dirs, sharex=True, sharey="row", figsize=(10, 8)
    )
    fig_E, ax_E = plt.subplots(3, num_dirs, sharex=True, sharey="row", figsize=(10, 8))
    fig_p, ax_p = plt.subplots()

    for idx, kappa in enumerate(resultdirs.keys()):
        title = "incomp" if kappa is None else rf"$\kappa = {kappa:.0f}$"
        ax_sigma[0, idx].set_title(title)
        ax_E[0, idx].set_title(title)

        lines = []
        labels = []
        for j, point in enumerate(points):
            for i, name in enumerate(["sigma_xx", "sigma_r", "sigma_c"]):
                (l,) = ax_sigma[i, idx].plot(
                    t, arrs[name][idx, :, j], color=cm.tab20(point / R)
                )
            for i, name in enumerate(["sigma_dev_xx", "sigma_dev_r", "sigma_dev_c"]):
                ax_sigma_dev[i, idx].plot(
                    t, arrs[name][idx, :, j], color=cm.tab20(point / R)
                )
            for i, name in enumerate(["E_xx", "E_r", "E_c"]):
                ax_E[i, idx].plot(t, arrs[name][idx, :, j], color=cm.tab20(point / R))

            if kappa is None:
                ax_p.set_title(title)
                ax_p.plot(t, arrs["p"][idx, :, j], color=cm.tab20(point / R))

            lines.append(l)
            labels.append(f"$r = {point:.1f}$")

    for i, label in enumerate([r"$\sigma_{xx}$", r"$\sigma_{r}$", r"$\sigma_{c}$"]):
        ax_sigma[i, 0].set_ylabel(label)

    for i, label in enumerate(
        [
            r"$\mathrm{dev}\sigma_{xx}$",
            r"$\mathrm{dev}\sigma_{r}$",
            r"$\mathrm{dev}\sigma_{c}$",
        ]
    ):
        ax_sigma_dev[i, 0].set_ylabel(label)

    for i, label in enumerate([r"$E_{xx}$", r"$E_{r}$", r"$E_{c}$"]):
        ax_E[i, 0].set_ylabel(label)

    ax_p2 = ax_p.set_ylabel("$p$")

    for fig in [fig_E, fig_sigma, fig_sigma_dev]:
        fig.subplots_adjust(right=0.87)
        fig.legend(lines, labels, loc="center right")
    fig_sigma.savefig(Path(figdir) / "results_comp_sigma.png")
    fig_sigma.savefig(Path(figdir) / "results_comp_sigma_dev.png")
    fig_E.savefig(Path(figdir) / "results_comp_E.png")
    fig_p.savefig(Path(figdir) / "results_comp_p.png")


def postprocess_effect_of_compressibility_disp(
    resultdirs, datadir="data_basic", figdir="figures_comp", plot_slice=True
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True)
    disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
    R = geo.mesh.coordinates().max(0)[-1]
    disk.coordinates()[:] *= R
    V_disk = dolfin.FunctionSpace(disk, "CG", 2)

    V_CG2 = dolfin.VectorFunctionSpace(geo.mesh, "CG", 2)
    u = dolfin.Function(V_CG2)
    u.set_allow_extrapolation(True)

    num_dirs = len(resultdirs)
    fst = next(iter(resultdirs.values()))
    t = np.load(Path(fst) / "t.npy")
    gamma = np.load(Path(fst) / "gamma.npy")
    N = len(t)

    points = np.linspace(0, R, 11)
    arr = np.zeros((num_dirs, N, len(points), 3))
    vols = np.zeros((num_dirs, N))
    mesh_vol = dolfin.assemble(dolfin.Constant(1) * dolfin.dx(geo.mesh))

    for idx, (kappa, resultdir) in enumerate(resultdirs.items()):
        output = Path(resultdir) / "results.xdmf"
        print(output)

        with dolfin.XDMFFile(output.as_posix()) as f:
            for i in range(N):
                f.read_checkpoint(u, "u", i)

                vols[idx, i] = (
                    dolfin.assemble(ufl.det(ufl.grad(u) + ufl.Identity(3)) * dolfin.dx)
                    / mesh_vol
                )

                for j, point in enumerate(points):
                    arr[idx, i, j, :] = u(0, 0, point)

                # if i == gamma.argmax() and plot_slice:
                #     for name, func in funcs.items():
                #         if name == "p" and kappa is not None:
                #             continue
                #         extra = (
                #             " (incomp)"
                #             if kappa is None
                #             else f" ($\\kappa={kappa:.0f}$)"
                #         )
                #         title = name2latex(name) + extra

                #         plt.colorbar(
                #             dolfin.plot(dolfin.interpolate(CylinderSlice(func), V_disk))
                #         )
                #         plt.title(title)
                #         plt.savefig(figdir / f"{name}_{kappa}.png")
                #         plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    fig_r, ax_r = plt.subplots(figsize=(10, 8))
    lines = []
    labels = []
    for idx, kappa in enumerate(resultdirs.keys()):
        title = "incomp" if kappa is None else rf"$\kappa = {kappa:.0f}$"
        labels.append(title)
        for i in range(3):
            (l,) = ax[i].plot(t, arr[idx, :, -1, i])
        y = arr[idx, :, -1, 1]
        z = arr[idx, :, -1, 2]
        dr = np.sqrt(y**2 + z**2)

        ax_r.plot(t, (R + dr) / R, label=title)
        lines.append(l)
    lgd = fig.legend(lines, labels, loc="center right")
    fig.subplots_adjust(right=0.87)
    ax_r.legend()
    ax_r.set_xlabel("time")
    ax_r.set_ylabel("radius change")

    for axi in ax:
        axi.set_xlabel("time")
    ax[0].set_ylabel("displacement")
    ax[0].set_title("$x$")
    ax[1].set_title("$y$")
    ax[2].set_title("$z$")
    fig_r.savefig(figdir / "results_comp_radius.png")
    fig.savefig(
        figdir / "results_comp_xyz.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, kappa in enumerate(resultdirs.keys()):
        title = "incomp" if kappa is None else rf"$\kappa = {kappa:.0f}$"
        ax.plot(t, vols[idx, :], label=title)
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("volume change")
    fig.savefig(figdir / "results_comp_volume.png")
