import dolfin
import cardiac_geometries
from pathlib import Path

import pulse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import ufl_legacy as ufl
from compressible_model import CompressibleProblem
import utils


dolfin.parameters["form_compiler"]["quadrature_degree"] = 6
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


def get_geometry():
    geofolder = Path("lv")
    if not geofolder.is_dir():
        # Assume rat heart has a short axis radius of 4 mm,
        # long axis raidus of 8 mm and a wall thickness of 1.5 mm
        cardiac_geometries.create_lv_ellipsoid(
            geofolder,
            r_short_endo=4.0,
            r_long_endo=8.0,
            r_short_epi=5.5,
            r_long_epi=9.5,
            psize_ref=1.0,
            create_fibers=True,
        )

    return cardiac_geometries.geometry.Geometry.from_folder(geofolder)


def load_arrs(arrs_path, output, gammas, pressures):
    geo = get_geometry()
    ds = dolfin.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)

    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 1)

    sigma_ff = dolfin.Function(V_DG1)
    sigma_ss = dolfin.Function(V_DG1)
    sigma_nn = dolfin.Function(V_DG1)

    sigma_dev_ff = dolfin.Function(V_DG1)
    sigma_dev_ss = dolfin.Function(V_DG1)
    sigma_dev_nn = dolfin.Function(V_DG1)

    E_ff = dolfin.Function(V_DG1)
    E_ss = dolfin.Function(V_DG1)
    E_nn = dolfin.Function(V_DG1)

    von_Mises = dolfin.Function(V_DG1)

    # u = dolfin.Function(dolfin.VectorFunctionSpace(geo.mesh, "CG", 2))
    # meshvol = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(geo.mesh))
    from postprocess import name2latex

    data = []
    with dolfin.XDMFFile(output.as_posix()) as xdmf:
        for ti in range(len(gammas)):
            # xdmf.read_checkpoint(u, "u", ti)
            for i, (func, name) in enumerate(
                zip(
                    [
                        sigma_ff,
                        sigma_ss,
                        sigma_nn,
                        E_ff,
                        E_ss,
                        E_nn,
                    ],
                    [
                        "sigma_ff",
                        "sigma_ss",
                        "sigma_nn",
                        "E_ff",
                        "E_ss",
                        "E_nn",
                    ],
                )
            ):
                xdmf.read_checkpoint(func, name, ti)
                f_arr = func.vector().get_local()
                # arrs[name][ti] = dolfin.assemble(func * dolfin.dx(geo.mesh)) / meshvol

                data.extend(
                    [
                        {
                            "name": name,
                            "value": fi,
                            "gamma": gammas[ti],
                            "pressure": pressures[ti],
                            "latex": name2latex(name),
                        }
                        for fi in f_arr
                    ]
                )
    df = pd.DataFrame(data)
    df.to_csv(output.with_suffix(".csv").as_posix())


def postprocess():
    resultsdir = Path("output_unloaded_ellipsoid")
    output = Path(resultsdir) / "results.xdmf"

    gammas = np.load(resultsdir / "gammas.npy")
    pressures = np.load(resultsdir / "pressures.npy")

    arrs_path = resultsdir / "arrs.npy"
    data_path = resultsdir / "results.csv"
    if not data_path.is_file():
        load_arrs(arrs_path, output, gammas, pressures)
    df = pd.read_csv(data_path)

    df_unloaded = df[
        np.isclose(df["pressure"], df["pressure"].min())
        & np.isclose(df["gamma"], df["gamma"].max())
    ]
    df_unloaded = df_unloaded.assign(label="unloaded")
    df_loaded = df[
        np.isclose(df["pressure"], df["pressure"].max())
        & np.isclose(df["gamma"], df["gamma"].max())
    ]
    df_loaded = df_loaded.assign(label="loaded")
    df1 = pd.concat([df_unloaded, df_loaded])

    df1_stress = df1[df1["name"].isin(["sigma_ff", "sigma_ss", "sigma_nn"])]
    fig = plt.figure()
    ax = sns.barplot(
        data=df1_stress,
        x="latex",
        y="value",
        hue="label",
        alpha=0.7,
    )
    sns.move_legend(
        ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("stress [kPa]")
    ax.grid()
    fig.savefig(resultsdir / "stress.png")  # type: ignore
    plt.close(fig)

    df1_strain = df1[df1["name"].isin(["E_ff", "E_ss", "E_nn"])]
    fig = plt.figure()
    ax = sns.barplot(
        data=df1_strain,
        x="latex",
        y="value",
        hue="label",
        alpha=0.7,
    )
    sns.move_legend(
        ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("strain")
    ax.grid()
    fig.savefig(resultsdir / "strain.png")  # type: ignore
    plt.close(fig)


def create_paraview_files():
    resultsdir = Path("output_unloaded_ellipsoid")
    gammas = np.load(resultsdir / "gammas.npy")
    output = Path(resultsdir) / "results.xdmf"
    pvd_output = Path(resultsdir) / "pvd_files"
    pvd_output.mkdir(exist_ok=True, parents=True)
    geo = get_geometry()

    #     moving_mesh = dolfin.Mesh(geo.mesh)
    # V_DG1_new = dolfin.FunctionSpace(moving_mesh, "DG", 1)
    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    V_CG2 = dolfin.VectorFunctionSpace(geo.mesh, "CG", 2)
    #     V_CG1 = dolfin.VectorFunctionSpace(geo.mesh, "CG", 1)
    u = dolfin.Function(V_CG2)
    u.rename("u", "")
    #     u_prev = dolfin.Function(V_CG2)
    #     u_diff = dolfin.Function(V_CG2)

    f = dolfin.Function(V_DG1)
    #     f_new = dolfin.Function(V_DG1_new)

    with dolfin.XDMFFile(output.as_posix()) as xdmf:
        for ti in range(len(gammas)):
            print(ti)
            xdmf.read_checkpoint(u, "u", ti)

            # u_diff.vector()[:] = u.vector()[:] - u_prev.vector()[:]
            # d = dolfin.interpolate(u_diff, V_CG1)
            # # dolfin.ALE.move(moving_mesh, d)
            # breakpoint()
            # # moving_mesh.coordinates()[:] += d.compute_vertex_values().reshape((-1, 3))
            # u_prev.vector()[:] = u.vector()[:]

            for i, name in enumerate(
                [
                    "sigma_ff",
                    "sigma_ss",
                    "sigma_nn",
                    # "E_ff",
                    # "E_ss",
                    # "E_nn",
                ],
            ):
                f.rename(name, "")
                xdmf.read_checkpoint(f, name, ti)
                with dolfin.XDMFFile(
                    (pvd_output / f"{name}_{ti}.xdmf").as_posix()
                ) as xdmf2:
                    xdmf2.parameters["functions_share_mesh"] = True
                    xdmf2.parameters["flush_output"] = True

                    xdmf2.write(u, ti)
                    xdmf2.write(f, ti)
                # f_new.vector()[:] = f.vector()[:]

                # dolfin.File(f"pvd_files/{name}_{ti}.pvd") << f_new


def main():
    # We only want standard output on rank 0. We therefore set the log level to
    # ERROR on all other ranks
    overwrite = True
    resultsdir = Path("output_unloaded_ellipsoid")
    Path(resultsdir).mkdir(exist_ok=True, parents=True)
    output = Path(resultsdir) / "results.xdmf"
    if output.is_file() and not overwrite:
        print(f"Output {output} already exists")
        return

    comm = dolfin.MPI.comm_world

    geo = get_geometry()

    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    proj = utils.Projector(V_DG1)
    sigma_ff = dolfin.Function(V_DG1)
    sigma_ss = dolfin.Function(V_DG1)
    sigma_nn = dolfin.Function(V_DG1)

    sigma_dev_ff = dolfin.Function(V_DG1)
    sigma_dev_ss = dolfin.Function(V_DG1)
    sigma_dev_nn = dolfin.Function(V_DG1)

    E_ff = dolfin.Function(V_DG1)
    E_ss = dolfin.Function(V_DG1)
    E_nn = dolfin.Function(V_DG1)

    von_Mises = dolfin.Function(V_DG1)

    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)

    geometry = pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        marker_functions=pulse.MarkerFunctions(ffun=geo.ffun),
        microstructure=microstructure,
    )

    m2mm = 1000.0

    matparams = {
        "a": 2280 / m2mm,
        "b": 9.726,
        "a_f": 1685 / m2mm,
        "b_f": 15.779,
        "a_s": 0.0,
        "b_s": 0.0,
        "a_fs": 0.0,
        "b_fs": 0.0,
    }
    gamma = dolfin.Constant(0.0)
    activation = gamma

    material = pulse.HolzapfelOgden(
        active_model="active_strain",
        activation=activation,
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    # Pericardium type Robin BC
    spring = dolfin.Constant(0.5)  # kPa/mm
    robin_bc = [
        pulse.RobinBC(value=dolfin.Constant(spring), marker=geo.markers["EPI"][0]),
    ]

    # LV Pressure
    lvp = dolfin.Constant(0.0)
    lv_marker = geometry.markers["ENDO"][0]
    lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
    neumann_bc = [lv_pressure]

    # Fix the basal plane in the longitudinal direction
    def fix_basal_plane(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = dolfin.DirichletBC(
            V.sub(0),
            dolfin.Constant(0.0),
            geometry.ffun,
            geometry.markers["BASE"][0],
        )
        return bc

    dirichlet_bc = [fix_basal_plane]

    bcs = pulse.BoundaryConditions(
        dirichlet=dirichlet_bc, neumann=neumann_bc, robin=robin_bc
    )

    problem = CompressibleProblem(geometry, material, bcs, kappa=1e3)
    gammas = [0.0, 0.3, 0.3]
    pressures = [0.0, 0.0, 15.0]

    np.save(resultsdir / "gammas.npy", gammas)
    np.save(resultsdir / "pressures.npy", pressures)

    for ti, (lvp_, g) in enumerate(zip(pressures, gammas)):
        print(lvp_, g)
        pulse.iterate.iterate(
            problem,
            (lvp, gamma),
            (lvp_, g),
            initial_number_of_steps=20,
            continuation=False,
        )
        u = problem.state

        F = pulse.kinematics.DeformationGradient(u)
        J = ufl.det(F)
        sigma = material.CauchyStress(F, p=None)
        sigma_dev = sigma - (1 / 3) * ufl.tr(sigma) * ufl.Identity(3)
        E = pulse.kinematics.GreenLagrangeStrain(F)
        f = F * geo.f0
        s = F * geo.s0
        n = F * geo.n0

        proj.project(von_Mises, utils.von_mises(sigma))

        proj.project(sigma_ff, dolfin.inner(f, sigma * f))
        proj.project(sigma_ss, dolfin.inner(s, sigma * s))
        proj.project(sigma_nn, dolfin.inner(n, sigma * n))

        proj.project(sigma_dev_ff, dolfin.inner(f, sigma_dev * f))
        proj.project(sigma_dev_ss, dolfin.inner(s, sigma_dev * s))
        proj.project(sigma_dev_nn, dolfin.inner(n, sigma_dev * n))

        proj.project(E_ff, dolfin.inner(geo.f0, E * geo.f0))
        proj.project(E_ss, dolfin.inner(geo.s0, E * geo.s0))
        proj.project(E_nn, dolfin.inner(geo.n0, E * geo.n0))
        with dolfin.XDMFFile(output.as_posix()) as f:
            f.parameters["functions_share_mesh"] = True
            f.parameters["rewrite_function_mesh"] = False
            f.write_checkpoint(
                u,
                function_name="u",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_ff,
                function_name="sigma_ff",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_ss,
                function_name="sigma_ss",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_nn,
                function_name="sigma_nn",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_dev_ff,
                function_name="sigma_dev_ff",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_dev_ss,
                function_name="sigma_dev_ss",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_dev_nn,
                function_name="sigma_dev_nn",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                E_ff,
                function_name="E_ff",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                E_ss,
                function_name="E_ss",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                E_nn,
                function_name="E_nn",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                von_Mises,
                function_name="von_Mises",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )


if __name__ == "__main__":
    # main()
    postprocess()
    # create_paraview_files()
