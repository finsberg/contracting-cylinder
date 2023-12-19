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


def create_arrays(resultdirs, gamma, points, figdir, plot_slice, geo, R):
    disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
    disk.coordinates()[:] *= R
    V_disk = dolfin.FunctionSpace(disk, "CG", 2)
    num_dirs = len(resultdirs)
    N = len(gamma)

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

    arrs = {name: np.zeros((num_dirs, N, len(points))) for name in funcs}
    arrs_avg = {name: np.zeros((num_dirs, N)) for name in funcs}

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

                for name, func in funcs.items():
                    arrs_avg[name][idx, i] = dolfin.assemble(func * dolfin.dx) / meshvol
                    for j, point in enumerate(points):
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

    np.save(figdir / "arrs.npy", {"arrs": arrs, "arrs_avg": arrs_avg})


def postprocess_effect_of_compressibility_stress(
    resultdirs, key2title, datadir="data_basic", figdir="figures_comp", plot_slice=True
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True)

    num_dirs = len(resultdirs)
    fst = next(iter(resultdirs.values()))
    t = np.load(Path(fst) / "t.npy")
    gamma = np.load(Path(fst) / "gamma.npy")
    N = len(t)
    R = geo.mesh.coordinates().max(0)[-1]
    points = np.linspace(0, R, 11)

    arr_path = Path(figdir) / "arrs.npy"
    if not arr_path.exists():
        create_arrays(resultdirs, gamma, points, figdir, plot_slice, geo, R)

    data_arrs = np.load(arr_path, allow_pickle=True).item()
    arrs = data_arrs["arrs"]
    arrs_avg = data_arrs["arrs_avg"]

    fig_sigma, ax_sigma = plt.subplots(
        3, num_dirs, sharex=True, sharey="row", figsize=(10, 8)
    )
    fig_sigma_dev, ax_sigma_dev = plt.subplots(
        3, num_dirs, sharex=True, sharey="row", figsize=(10, 8)
    )
    fig_E, ax_E = plt.subplots(3, num_dirs, sharex=True, sharey="row", figsize=(10, 8))
    fig_p, ax_p = plt.subplots()

    fig_E_avg, ax_E_avg = plt.subplots(3, 1, sharex=True, sharey="row", figsize=(6, 8))
    fig_p_avg, ax_p_avg = plt.subplots()

    fig_sigma_avg, ax_sigma_avg = plt.subplots(
        3, 1, sharex=True, sharey="row", figsize=(6, 8)
    )
    fig_sigma_dev_avg, ax_sigma_dev_avg = plt.subplots(
        3, 1, sharex=True, sharey="row", figsize=(6, 8)
    )

    lines_avg = []
    labels_avg = []
    lines = []
    labels = []
    for idx, key in enumerate(resultdirs.keys()):
        title = key2title(key)

        ax_sigma[0, idx].set_title(title)
        ax_E[0, idx].set_title(title)

        for k, (i, name, ax, ax_avg) in enumerate(
            [
                (0, "sigma_xx", ax_sigma, ax_sigma_avg),
                (1, "sigma_r", ax_sigma, ax_sigma_avg),
                (2, "sigma_c", ax_sigma, ax_sigma_avg),
                (0, "sigma_dev_xx", ax_sigma_dev, ax_sigma_dev_avg),
                (1, "sigma_dev_r", ax_sigma_dev, ax_sigma_dev_avg),
                (2, "sigma_dev_c", ax_sigma_dev, ax_sigma_dev_avg),
                (0, "E_xx", ax_E, ax_E_avg),
                (1, "E_r", ax_E, ax_E_avg),
                (2, "E_c", ax_E, ax_E_avg),
                (0, "p", ax_p, ax_p_avg),
            ]
        ):
            if name == "p":
                axi_avg = ax_avg
            else:
                axi_avg = ax_avg[i]
            (l1,) = axi_avg.plot(
                t, arrs_avg[name][idx, :], color=cm.tab20(idx / len(resultdirs))
            )
            if k == 0:
                lines_avg.append(l1)
                labels_avg.append(title)
            for j, point in enumerate(points):
                if name == "p":
                    if key is not None:
                        continue
                    axi = ax
                else:
                    axi = ax[i, idx]
                (l,) = axi.plot(t, arrs[name][idx, :, j], color=cm.tab20(point / R))
                if k == 0 and idx == 0:
                    lines.append(l)
                    labels.append(f"$r = {point:.1f}$")

    for i, label in enumerate([r"$\sigma_{xx}$", r"$\sigma_{r}$", r"$\sigma_{c}$"]):
        ax_sigma[i, 0].set_ylabel(label)
        ax_sigma_avg[i].set_title(label)

    for i, label in enumerate(
        [
            r"$\mathrm{dev}\sigma_{xx}$",
            r"$\mathrm{dev}\sigma_{r}$",
            r"$\mathrm{dev}\sigma_{c}$",
        ]
    ):
        ax_sigma_dev[i, 0].set_ylabel(label)
        ax_sigma_dev_avg[i].set_title(label)

    for i, label in enumerate([r"$E_{xx}$", r"$E_{r}$", r"$E_{c}$"]):
        ax_E[i, 0].set_ylabel(label)
        ax_E_avg[i].set_title(label)

    ax_p2 = ax_p.set_ylabel("$p$")

    for fig in [fig_E, fig_sigma, fig_sigma_dev]:
        fig.subplots_adjust(right=0.87)
        fig.legend(lines, labels, loc="center right")

    for fig in [fig_E_avg, fig_sigma_avg, fig_sigma_dev_avg]:
        fig.subplots_adjust(right=0.75)
        fig.legend(lines_avg, labels_avg, loc="center right")
    fig_sigma.savefig(Path(figdir) / "results_comp_sigma.png")
    fig_sigma_dev.savefig(Path(figdir) / "results_comp_sigma_dev.png")
    fig_E.savefig(Path(figdir) / "results_comp_E.png")
    fig_p.savefig(Path(figdir) / "results_comp_p.png")
    fig_sigma_avg.savefig(Path(figdir) / "results_comp_sigma_avg.png")
    fig_sigma_dev_avg.savefig(Path(figdir) / "results_comp_sigma_dev_avg.png")
    fig_E_avg.savefig(Path(figdir) / "results_comp_E_avg.png")
    fig_p_avg.savefig(Path(figdir) / "results_comp_p_avg.png")


def postprocess_effect_of_compressibility_disp(
    resultdirs,
    key2title,
    datadir="data_basic",
    figdir="figures_comp",
    plot_slice=True,
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True)
    disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
    R = geo.mesh.coordinates().max(0)[-1]
    L = geo.mesh.coordinates().max(0)[0]

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
    arr_center = np.zeros((num_dirs, N, 3))
    arr_long = np.zeros((num_dirs, N, 3))
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

                arr_center[idx, i, :] = u(0, 0, R)
                arr_long[idx, i, :] = u(L, 0, 0)

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

    fig, ax = plt.subplots()
    x = ["Relaxation"] + [f"$\\kappa={k:.0f}$" for k in resultdirs.keys()]
    y_d = [2 * R] + [
        2 * (R + np.sqrt(arr_center[idx, 1, 1] ** 2 + arr_center[idx, 1, 2] ** 2))
        for idx in range(num_dirs)
    ]
    ax.bar(x, y_d)
    ax.set_ylabel("Diameter")
    ax.grid()
    fig.savefig(figdir / "results_bar_diameter.png")

    fig, ax = plt.subplots()
    x = ["Relaxation"] + [f"$\\kappa={k:.0f}$" for k in resultdirs.keys()]
    y_l = [2 * L] + [2 * (L + arr_long[idx, 1, 0]) for idx in range(num_dirs)]
    ax.bar(x, y_l)
    ax.set_ylabel("Length")
    ax.grid()
    fig.savefig(figdir / "results_bar_length.png")

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    fig_d, ax_d = plt.subplots(figsize=(10, 8))
    fig_long, ax_long = plt.subplots(figsize=(10, 8))

    lines = []
    labels = []
    for idx, kappa in enumerate(resultdirs.keys()):
        title = key2title(kappa)

        labels.append(title)
        for i in range(3):
            (l,) = ax[i].plot(t, arr_center[idx, :, i])
        y = arr_center[idx, :, 1]
        z = arr_center[idx, :, 2]
        dr = np.sqrt(y**2 + z**2)

        ax_d.plot(t, 2 * (R + dr), label=title)
        ax_long.plot(t, 2 * (L + arr_long[idx, :, 0]), label=title)

        lines.append(l)
    lgd = fig.legend(lines, labels, loc="center right")
    fig.subplots_adjust(right=0.87)
    ax_d.legend()
    ax_d.set_xlabel("time")
    ax_d.set_ylabel("diameter")

    ax_long.legend()
    ax_long.set_xlabel("time")
    ax_long.set_ylabel("length")

    for axi in ax:
        axi.set_xlabel("time")
    ax[0].set_ylabel("displacement")
    ax[0].set_title("$x$")
    ax[1].set_title("$y$")
    ax[2].set_title("$z$")
    fig_d.savefig(figdir / "results_comp_diameter.png")
    fig_long.savefig(figdir / "results_comp_length.png")
    fig.savefig(
        figdir / "results_comp_xyz.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, kappa in enumerate(resultdirs.keys()):
        title = key2title(kappa)
        ax.plot(t, vols[idx, :], label=title)
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("volume change")
    fig.savefig(figdir / "results_comp_volume.png")
