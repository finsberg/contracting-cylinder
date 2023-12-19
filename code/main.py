from pathlib import Path
import dolfin
import pulse
import numpy as np
import ufl_legacy as ufl


from geometry import load_geometry
from utils import Projector
import postprocess
from compressible_model import CompressibleProblem


def ca_transient(t, tstart=0.05, ca_ampl=0.3):
    tau1 = 0.05
    tau2 = 0.110

    ca_diast = 0.0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (
        -1 / (1 - tau2 / tau1)
    )
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1)
        - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca


def run(
    resultsdir="results",
    datadir="data",
    spring=0.1,
    target_preload=0.0,
    overwrite: bool = False,
    varying_gamma: bool = False,
    kappa: float | None = None,
):
    pulse.set_log_level(10)
    Path(resultsdir).mkdir(exist_ok=True, parents=True)
    output = Path(resultsdir) / "results.xdmf"
    if output.is_file() and not overwrite:
        print(f"Output {output} already exists")
        return
    output.unlink(missing_ok=True)
    output.with_suffix(".h5").unlink(missing_ok=True)

    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")._asdict()

    microstructure = pulse.Microstructure(
        f0=dolfin.as_vector([1.0, 0.0, 0.0]),
        s0=dolfin.as_vector([0.0, 1.0, 0.0]),
        n0=dolfin.as_vector([0.0, 0.0, 1.0]),
    )

    geometry = pulse.Geometry(
        **geo,
        microstructure=microstructure,
    )

    gamma = dolfin.Constant(0.0)
    if varying_gamma:
        V_CG1 = dolfin.FunctionSpace(geometry.mesh, "CG", 1)
        gamma_expr = dolfin.Expression("x[1]*x[1] + x[2]*x[2]", degree=2)
        activation = dolfin.Function(V_CG1)
        activation.interpolate(gamma_expr)
        activation *= gamma
    else:
        activation = gamma

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

    # matparams = pulse.HolzapfelOgden.default_parameters()
    material = pulse.HolzapfelOgden(
        active_model="active_strain",
        activation=activation,
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    # matparams = pulse.Guccione.default_parameters()
    # material = pulse.Guccione(
    #     active_model="active_strain",
    #     activation=activation,
    #     parameters=matparams,
    #     f0=geometry.f0,
    #     s0=geometry.s0,
    #     n0=geometry.n0,
    # )

    robin_bc = [
        pulse.RobinBC(
            value=dolfin.Constant(spring), marker=geometry.markers["Right"][0]
        ),
        pulse.RobinBC(
            value=dolfin.Constant(spring), marker=geometry.markers["Left"][0]
        ),
    ]

    preload = dolfin.Constant(0.0)

    neumann_bc = [
        pulse.NeumannBC(traction=preload, marker=geometry.markers["Right"][0]),
        pulse.NeumannBC(traction=-preload, marker=geometry.markers["Left"][0]),
    ]

    bcs = pulse.BoundaryConditions(
        dirichlet=[lambda x: []], neumann=neumann_bc, robin=robin_bc
    )

    if kappa is None:
        problem = pulse.MechanicsProblem(geometry, material, bcs)
    else:
        problem = CompressibleProblem(geometry, material, bcs, kappa=kappa)

    W_DG1 = dolfin.VectorFunctionSpace(geometry.mesh, "DG", 1)
    rad0 = dolfin.Function(W_DG1)
    rad0.interpolate(
        dolfin.Expression(
            (
                "0",
                "x[1]/sqrt(x[1]*x[1] + x[2]*x[2])",
                "x[2]/sqrt(x[1]*x[1] + x[2]*x[2])",
            ),
            degree=1,
        )
    )
    circ0 = dolfin.Function(W_DG1)
    circ0.interpolate(
        dolfin.Expression(
            (
                "0",
                "x[2]/sqrt(x[1]*x[1] + x[2]*x[2])",
                "-x[1]/sqrt(x[1]*x[1] + x[2]*x[2])",
            ),
            degree=1,
        )
    )
    long0 = dolfin.Function(W_DG1)
    long0.interpolate(dolfin.Expression(("1", "0", "0"), degree=1))

    V_DG1 = dolfin.FunctionSpace(geometry.mesh, "DG", 1)
    proj = Projector(V_DG1)
    sigma_xx = dolfin.Function(V_DG1)
    sigma_r = dolfin.Function(V_DG1)
    sigma_c = dolfin.Function(V_DG1)

    sigma_dev_xx = dolfin.Function(V_DG1)
    sigma_dev_r = dolfin.Function(V_DG1)
    sigma_dev_c = dolfin.Function(V_DG1)

    E_xx = dolfin.Function(V_DG1)
    E_r = dolfin.Function(V_DG1)
    E_c = dolfin.Function(V_DG1)

    N = 5
    t = np.linspace(0, 1, N)
    amp = ca_transient(t, ca_ampl=0.3)
    t = [0, 0.2, 1.0]
    amp = [0.0, 0.3, 0.0]
    # breakpoint()
    np.save(Path(resultsdir) / "t.npy", t)
    np.save(Path(resultsdir) / "gamma.npy", amp)

    if not np.isclose(target_preload, 0.0):
        pulse.iterate.iterate(
            problem, preload, target_preload, initial_number_of_steps=20
        )
        with dolfin.XDMFFile(output.as_posix()) as f:
            f.write_checkpoint(
                problem.state,
                function_name="u",
                time_step=0.0,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )

    for i, (ti, g) in enumerate(zip(t, amp)):
        pulse.iterate.iterate(problem, gamma, g, initial_number_of_steps=20)

        if kappa is None:
            u, p = problem.state.split()
        else:
            u = problem.state
            p = None

        F = pulse.kinematics.DeformationGradient(u)
        J = ufl.det(F)
        sigma = material.CauchyStress(F, p)
        sigma_dev = sigma - (1 / 3) * ufl.tr(sigma) * ufl.Identity(3)
        E = pulse.kinematics.GreenLagrangeStrain(F)
        rad = F * rad0
        circ = F * circ0
        long = F * long0

        proj.project(sigma_xx, dolfin.inner(long, sigma * long))
        proj.project(sigma_r, dolfin.inner(rad, sigma * rad))
        proj.project(sigma_c, dolfin.inner(circ, sigma * circ))

        proj.project(sigma_dev_xx, dolfin.inner(long, sigma_dev * long))
        proj.project(sigma_dev_r, dolfin.inner(rad, sigma_dev * rad))
        proj.project(sigma_dev_c, dolfin.inner(circ, sigma_dev * circ))

        proj.project(E_xx, dolfin.inner(long0, E * long0))
        proj.project(E_r, dolfin.inner(rad0, E * rad0))
        proj.project(E_c, dolfin.inner(circ0, E * circ0))
        with dolfin.XDMFFile(output.as_posix()) as f:
            f.write_checkpoint(
                u,
                function_name="u",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            if p is not None:
                f.write_checkpoint(
                    p,
                    function_name="p",
                    time_step=ti,
                    encoding=dolfin.XDMFFile.Encoding.HDF5,
                    append=True,
                )
            f.write_checkpoint(
                sigma_xx,
                function_name="sigma_xx",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_r,
                function_name="sigma_r",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_c,
                function_name="sigma_c",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_dev_xx,
                function_name="sigma_dev_xx",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_dev_r,
                function_name="sigma_dev_r",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                sigma_dev_c,
                function_name="sigma_dev_c",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                E_xx,
                function_name="E_xx",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                E_r,
                function_name="E_r",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                E_c,
                function_name="E_c",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )


def run_basic():
    run(resultsdir="results/basic", datadir="data_basic", overwrite=True)
    postprocess.postprocess_basic(
        resultsdir="results/basic", datadir="data_basic", figdir="figures_new/basic"
    )


def run_basic_smaller_raidius():
    # run(
    #     datadir="data_small_radius", resultsdir="results_smaller_radius", overwrite=True
    # )
    postprocess.postprocess_basic(
        datadir="data_small_radius",
        resultsdir="results_smaller_radius",
        figdir="figures_smaller_radius",
    )


def run_varing_gamma():
    run(resultsdir="results_varying_gamma", varying_gamma=True, overwrite=True)
    postprocess.postprocess_basic(
        resultsdir="results_varying_gamma", figdir="figures_varying_gamma"
    )


def effect_of_spring():
    springs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    resultdirs = {spring: f"results/spring{spring}" for spring in springs}
    for spring in springs:
        print("Spring : ", spring)
        run(resultsdir=resultdirs[spring], spring=spring, datadir="data_basic")
    postprocess.postprocess_effect_of_spring(resultdirs=resultdirs)


def run_small():
    import create_mesh

    datadir = "data_small"
    create_mesh.main(datadir=datadir, char_length=1.0)
    run(datadir=datadir, resultsdir="results_small", overwrite=True, spring=10.0)
    print("Done")
    postprocess.postprocess_basic(resultsdir="results_small", datadir=datadir)


def run_compressiblity():
    kappas = [1, 10, 1e2, 1e3, 1e4]
    resultdirs = {kappa: f"results/kappa{kappa}" for kappa in kappas}
    # for kappa in kappas:
    #     run(
    #         resultsdir=resultdirs[kappa],
    #         overwrite=True,
    #         datadir="data_basic",
    #         kappa=kappa,
    #     )
    # print("Done")
    # exit()
    key2title = lambda kappa: "incomp" if kappa is None else rf"$\kappa = {kappa:.0f}$"
    # postprocess.postprocess_effect_of_compressibility_stress(
    # resultdirs=resultdirs, key2title=key2title
    # )
    postprocess.postprocess_effect_of_compressibility_disp(
        resultdirs=resultdirs, key2title=key2title
    )


def run_basic_with_preload():
    preloads = [-10, -8, -6, -4, -2, 0]
    resultdirs = {
        preload: f"results/basic_with_preload_{preload}" for preload in preloads
    }
    for preload in preloads:
        run(
            resultsdir=resultdirs[preload],
            datadir="data_basic",
            overwrite=True,
            target_preload=preload,
            kappa=1e3,
        )

    key2title = lambda preload: rf"$F = {preload:.0f}$"
    postprocess.postprocess_effect_of_compressibility_stress(
        resultdirs=resultdirs, figdir="figures_preload", key2title=key2title
    )


def main():
    run_basic_with_preload()
    # run_compressiblity()
    # run_basic()
    # effect_of_spring()
    # run_varing_gamma()
    # run_small()
    # run_basic_smaller_raidius()


if __name__ == "__main__":
    main()
