import dolfin
import cardiac_geometries
from pathlib import Path

import pulse
import matplotlib.pyplot as plt


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


def load_arrs(arrs_path, output, N):
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
    meshvol = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(geo.mesh))
    endo_surface = dolfin.assemble(dolfin.Constant(1.0) * ds(geo.markers["ENDO"][0]))
    epi_surface = dolfin.assemble(dolfin.Constant(1.0) * ds(geo.markers["EPI"][0]))
    names = [
        "sigma_ff",
        "sigma_ss",
        "sigma_nn",
        "sigma_dev_ff",
        "sigma_dev_ss",
        "sigma_dev_nn",
        "E_ff",
        "E_ss",
        "E_nn",
        "von_Mises",
    ]
    arrs = {name: np.zeros(N) for name in names}
    arrs_endo = {name: np.zeros(N) for name in names}
    arrs_epi = {name: np.zeros(N) for name in names}

    with dolfin.XDMFFile(output.as_posix()) as xdmf:
        for ti in range(N):
            # xdmf.read_checkpoint(u, "u", ti)
            for i, (func, name) in enumerate(
                zip(
                    [
                        sigma_ff,
                        sigma_ss,
                        sigma_nn,
                        sigma_dev_ff,
                        sigma_dev_ss,
                        sigma_dev_nn,
                        E_ff,
                        E_ss,
                        E_nn,
                        von_Mises,
                    ],
                    [
                        "sigma_ff",
                        "sigma_ss",
                        "sigma_nn",
                        "sigma_dev_ff",
                        "sigma_dev_ss",
                        "sigma_dev_nn",
                        "E_ff",
                        "E_ss",
                        "E_nn",
                        "von_Mises",
                    ],
                )
            ):
                xdmf.read_checkpoint(func, name, ti)
                arrs[name][ti] = dolfin.assemble(func * dolfin.dx(geo.mesh)) / meshvol
                arrs_endo[name][ti] = (
                    dolfin.assemble(func * ds(geo.markers["ENDO"][0])) / endo_surface
                )
                arrs_epi[name][ti] = (
                    dolfin.assemble(func * ds(geo.markers["EPI"][0])) / epi_surface
                )

    np.save(arrs_path, {"avg": arrs, "endo": arrs_endo, "epi": arrs_epi})


def postprocess():
    resultsdir = Path("output_unloaded_ellipsoid")
    output = Path(resultsdir) / "results.xdmf"

    gammas = np.load(resultsdir / "gammas.npy")
    pressures = np.load(resultsdir / "pressures.npy")
    N = len(gammas)

    arrs_path = resultsdir / "arrs.npy"
    if not arrs_path.is_file():
        load_arrs(arrs_path, output, N)
    arrs_data = np.load(arrs_path, allow_pickle=True).item()
    arrs = arrs_data["avg"]
    arrs_endo = arrs_data["endo"]
    arrs_epi = arrs_data["epi"]

    no_pressure_inds = np.arange(0, 5, dtype=int)
    pressure_inds = np.arange(5, N, dtype=int)
    inds = np.arange(N, dtype=int)

    # breakpoint()
    for dev in ["", "_dev"]:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        for i, (arrs_, label) in enumerate(
            zip([arrs, arrs_endo, arrs_epi], ["avg", "endo", "epi"])
        ):
            (l1,) = ax[i].plot(inds, arrs_[f"sigma{dev}_ff"])
            (l2,) = ax[i].plot(inds, arrs_[f"sigma{dev}_ss"])
            (l3,) = ax[i].plot(inds, arrs_[f"sigma{dev}_nn"])
            ax[i].grid()
            ax[i].set_xticks(inds)
            ax[i].set_xticklabels([])
            ax[i].set_ylabel(f"stress[kPa] ({label})")
            ax[i].set_xlim(inds[0], inds[-1])

        # breakpoint()

        ax2 = ax[0].twiny()
        ax2.set_xticks(inds)
        ax2.set_xticklabels(pressures)
        ax[-1].set_xticklabels(gammas)
        ax[-1].set_xlabel("gamma (activation)")
        ax2.set_xlabel("pressure")

        lines = (l1, l2, l3)
        labels = ["ff", "ss", "nn"]
        # fig.subplots_adjust(top=0.85)
        lgd = fig.legend(lines, labels, loc="upper center", ncol=3)
        fig.savefig(
            resultsdir / f"sigma{dev}.png",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )

    fig, ax = plt.subplots()
    ax.plot(inds, arrs["E_ff"], label="ff")
    ax.plot(inds, arrs["E_ss"], label="ss")
    ax.plot(inds, arrs["E_nn"], label="nn")
    ax.set_xticks(inds)
    ax.set_xticklabels(gammas)
    ax.grid()
    ax2 = ax.twiny()
    ax2.set_xticks(inds)
    ax2.set_xticklabels(pressures)
    ax2.set_xlabel("pressure")
    ax.set_xlabel("gamma")
    ax.set_ylabel("strain")
    ax.legend()
    fig.savefig(resultsdir / "strain.png")


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

    # Read geometry from file. If the file is not present we regenerate it.
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
    # gammas = [0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1, 0.0]
    # pressures = [0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 20.0, 20.0, 20.0]
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
