from pathlib import Path
import dolfin
import pulse
import numpy as np


from geometry import load_geometry
from utils import Projector
import postprocess


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
    spring=1.0,
    overwrite: bool = False,
    varying_gamma: bool = False,
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

    matparams = {
        "a": 2.280,
        "b": 9.726,
        "a_f": 1.685,
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
    bcs = pulse.BoundaryConditions(dirichlet=[lambda x: []], neumann=[], robin=robin_bc)

    problem = pulse.MechanicsProblem(geometry, material, bcs)

    W_DG1 = dolfin.VectorFunctionSpace(geometry.mesh, "DG", 1)
    rad = dolfin.Function(W_DG1)
    rad.interpolate(
        dolfin.Expression(
            (
                "0",
                "x[1]/sqrt(x[1]*x[1] + x[2]*x[2])",
                "x[2]/sqrt(x[1]*x[1] + x[2]*x[2])",
            ),
            degree=1,
        )
    )
    circ = dolfin.Function(W_DG1)
    circ.interpolate(
        dolfin.Expression(
            (
                "0",
                "x[2]/sqrt(x[1]*x[1] + x[2]*x[2])",
                "-x[1]/sqrt(x[1]*x[1] + x[2]*x[2])",
            ),
            degree=1,
        )
    )

    V_DG1 = dolfin.FunctionSpace(geometry.mesh, "DG", 2)
    proj = Projector(V_DG1)
    sigma_xx = dolfin.Function(V_DG1)
    sigma_r = dolfin.Function(V_DG1)
    sigma_c = dolfin.Function(V_DG1)
    E_xx = dolfin.Function(V_DG1)
    E_r = dolfin.Function(V_DG1)
    E_c = dolfin.Function(V_DG1)

    N = 50
    t = np.linspace(0, 1, N)
    amp = ca_transient(t, ca_ampl=0.2)
    np.save(Path(resultsdir) / "t.npy", t)
    np.save(Path(resultsdir) / "gamma.npy", amp)

    for i, (ti, g) in enumerate(zip(t, amp)):
        pulse.iterate.iterate(problem, gamma, g, initial_number_of_steps=20)
        u, p = problem.state.split()
        F = pulse.kinematics.DeformationGradient(u)
        sigma = material.CauchyStress(F, p)
        E = pulse.kinematics.GreenLagrangeStrain(F)

        proj.project(sigma_xx, sigma[0, 0])
        proj.project(sigma_r, dolfin.inner(rad, sigma * rad))
        proj.project(sigma_c, dolfin.inner(circ, sigma * circ))
        proj.project(E_xx, E[0, 0])
        proj.project(E_r, dolfin.inner(rad, E * rad))
        proj.project(E_c, dolfin.inner(circ, E * circ))

        with dolfin.XDMFFile(output.as_posix()) as f:
            f.write_checkpoint(
                u,
                function_name="u",
                time_step=ti,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
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
    run(overwrite=True)
    postprocess.postprocess_basic()


def run_varing_gamma():
    run(resultsdir="results_varying_gamma", varying_gamma=True, overwrite=True)
    postprocess.postprocess_basic(
        resultsdir="results_varying_gamma", figdir="figures_varying_gamma"
    )


def effect_of_spring():
    springs = [0.5, 1.0, 5.0, 10.0]
    resultdirs = {spring: f"results_spring{spring}" for spring in springs}
    for spring in springs:
        print("Spring : ", spring)
        run(resultsdir=resultdirs[spring], spring=spring)
    postprocess.postprocess_effect_of_spring(resultdirs=resultdirs)


def run_small():
    import create_mesh

    datadir = "data_small"
    create_mesh.main(datadir=datadir, char_length=1.0)
    run(datadir=datadir, resultsdir="results_small", overwrite=True, spring=10.0)
    print("Done")
    postprocess.postprocess_basic(resultsdir="results_small", datadir=datadir)


def main():
    # run_basic()
    effect_of_spring()
    run_varing_gamma()
    # run_small()


if __name__ == "__main__":
    main()
