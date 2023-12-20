from pathlib import Path
import dolfin
import pandas as pd
import seaborn as sns

import ufl_legacy as ufl
import numpy as np
import matplotlib.pyplot as plt

from geometry import load_geometry


class CylinderSlice(dolfin.UserExpression):
    def __init__(self, f):
        self.f = f
        super().__init__()

    def eval(self, value, x):
        dx = 5.0
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


def create_arrays(
    resultdirs, data_file, gamma, figdir, geo, key2title, plot_slice=False
):
    R = geo.mesh.coordinates().max(0)[-1]
    L = geo.mesh.coordinates().max(0)[0]
    if plot_slice:
        disk = dolfin.UnitDiscMesh.create(dolfin.MPI.comm_world, 30, 1, 2)
        disk.coordinates()[:] *= R
        V_disk = dolfin.FunctionSpace(disk, "DG", 1)

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

    data = []

    for idx, (key, resultdir) in enumerate(resultdirs.items()):
        output = Path(resultdir) / "results.xdmf"
        print(output)

        with dolfin.XDMFFile(output.as_posix()) as f:
            for i in range(N):
                for name, func in funcs.items():
                    print(name)
                    if name == "p" and key is not None:
                        continue

                    f.read_checkpoint(func, name, i)

                    title = name2latex(name) + key2title(key)

                    # Average over the dofs in the center
                    dofs = np.abs(V_DG1.tabulate_dof_coordinates()[:, 0]) < 0.1 * L
                    f_arr = func.vector().get_local()[dofs]
                    # avg = np.mean(f_arr)
                    # std = np.std(f_arr)

                    # avg = dolfin.assemble((func / meshvol) * geo.dx(1))
                    # std = np.sqrt(
                    # dolfin.assemble(((func / meshvol) - avg) ** 2 * geo.dx(1))
                    # )
                    # arrs_avg[name][idx, i] = avg
                    # arrs_std[name][idx, i] = std

                    data.extend(
                        [
                            {
                                "name": name,
                                "key": key,
                                "value": fi,
                                "gamma": gamma[i],
                                "latex": name2latex(name),
                                "xlabel": key2title(key),
                            }
                            for fi in f_arr
                        ]
                    )

                    if plot_slice:
                        f_int = dolfin.plot(
                            dolfin.interpolate(CylinderSlice(func), V_disk)
                        )
                        plt.colorbar(f_int)
                        plt.title(title)
                        plt.savefig(figdir / f"{name}_{key}_{i}.png")
                        plt.close()
    df = pd.DataFrame(data)
    df.to_csv(data_file)


def postprocess_stress(
    resultdirs, key2title, datadir="data_basic", figdir="figures_comp", plot_slice=False
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True)

    fst = next(iter(resultdirs.values()))
    t = np.load(Path(fst) / "t.npy")
    gamma = np.load(Path(fst) / "gamma.npy")

    data_file = figdir / "data.csv"
    if not data_file.exists():
        create_arrays(
            resultdirs=resultdirs,
            gamma=gamma,
            figdir=figdir,
            plot_slice=plot_slice,
            geo=geo,
            key2title=key2title,
            data_file=data_file,
        )

    # data_arrs = np.load(arr_path, allow_pickle=True).item()
    # arrs_std = data_arrs["arrs_std"]
    # arrs_avg = data_arrs["arrs_avg"]
    df = pd.read_csv(data_file)
    max_df = df[np.isclose(df["gamma"], df["gamma"].max())]
    stress_df = max_df[max_df["name"].isin(["sigma_xx", "sigma_r", "sigma_c"])]

    fig = plt.figure()
    ax = sns.barplot(
        data=stress_df,
        x="xlabel",
        y="value",
        hue="latex",
        errorbar="ci",
        alpha=0.7,
    )
    ax.get_legend().set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Average stress")
    fig.savefig(figdir / "stress.png")

    fig = plt.figure()
    ax = sns.barplot(
        data=stress_df,
        x="xlabel",
        y="value",
        hue="latex",
        errorbar="sd",
        alpha=0.7,
    )
    ax.get_legend().set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Average stress")
    fig.savefig(figdir / "stress_yerr.png")

    ax.set_yscale("log")
    fig.savefig(figdir / "stress_log.png")
    plt.close(fig)

    fig = plt.figure()
    strain_df = max_df[max_df["name"].isin(["E_xx", "E_r", "E_c"])]
    ax = sns.barplot(
        data=strain_df,
        x="xlabel",
        y="value",
        hue="latex",
        alpha=0.7,
    )
    ax.get_legend().set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Average strain")
    fig.savefig(figdir / "strain.png")
    plt.close(fig)


def postprocess_disp(
    resultdirs,
    key2title,
    datadir="data_basic",
    figdir="figures_comp",
):
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True)

    R = geo.mesh.coordinates().max(0)[-1]
    L = geo.mesh.coordinates().max(0)[0]

    V_CG2 = dolfin.VectorFunctionSpace(geo.mesh, "CG", 2)
    u = dolfin.Function(V_CG2)
    u.set_allow_extrapolation(True)

    num_dirs = len(resultdirs)
    fst = next(iter(resultdirs.values()))
    t = np.load(Path(fst) / "t.npy")
    N = len(t)

    # arr_center = np.zeros((num_dirs, N, 3))
    # arr_long = np.zeros((num_dirs, N, 3))
    vols = np.zeros((num_dirs, N))
    mesh_vol = dolfin.assemble(dolfin.Constant(1) * dolfin.dx(geo.mesh))

    data_path = figdir / "data_u.csv"
    if not data_path.is_file():
        data = []
        for idx, (key, resultdir) in enumerate(resultdirs.items()):
            output = Path(resultdir) / "results.xdmf"
            print(output)

            with dolfin.XDMFFile(output.as_posix()) as f:
                for i in range(N):
                    f.read_checkpoint(u, "u", i)

                    vols[idx, i] = (
                        dolfin.assemble(
                            ufl.det(ufl.grad(u) + ufl.Identity(3)) * dolfin.dx
                        )
                        / mesh_vol
                    )
                    u_center = u(0, 0, R)
                    u_long = u(L, 0, 0)
                    data.append(
                        {
                            "u_center_x": u_center[0],
                            "u_center_y": u_center[1],
                            "u_center_z": u_center[2],
                            "u_long_x": u_long[0],
                            "u_long_y": u_long[1],
                            "u_long_z": u_long[2],
                            "length": 2 * (L + u_long[0]),
                            "diameter": 2
                            * (R + np.sqrt(u_center[1] ** 2 + u_center[2] ** 2)),
                            "key": key,
                            "volume": vols[idx, i],
                            "gamma": t[i],
                            "title": key2title(key),
                            "time_point": i,
                            "label": "Contaction" if i == 1 else "Relaxation",
                        }
                    )
        df = pd.DataFrame(data)
        df.to_csv(data_path)
    df = pd.read_csv(data_path)
    df_rc = df[df["time_point"].isin([0, 1])]

    # breakpoint()

    fig = plt.figure()
    ax = sns.barplot(
        data=df_rc,
        x="title",
        y="length",
        hue="label",
        alpha=0.7,
    )
    sns.move_legend(
        ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("Length [mm]")
    ax.grid()
    fig.savefig(figdir / "length.png")
    plt.close(fig)

    fig = plt.figure()
    ax = sns.barplot(
        data=df_rc,
        x="title",
        y="volume",
        hue="label",
        alpha=0.7,
    )
    sns.move_legend(
        ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("Volume change")
    fig.savefig(figdir / "volume.png")
    plt.close(fig)

    fig = plt.figure()
    ax = sns.barplot(
        data=df_rc,
        x="title",
        y="diameter",
        hue="label",
        alpha=0.7,
    )
    sns.move_legend(
        ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("Diameter [mm]")
    fig.savefig(figdir / "diameter.png")
    plt.close(fig)
