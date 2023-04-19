from pathlib import Path
import dolfin
import pulse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py

from geometry import load_geometry
from utils import Projector


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


def main(resultsdir="results", datadir="data"):
    pulse.set_log_level(10)

    microstructure = pulse.Microstructure(
        f0=dolfin.as_vector([1.0, 0.0, 0.0]),
        s0=dolfin.as_vector([0.0, 1.0, 0.0]),
        n0=dolfin.as_vector([0.0, 0.0, 1.0]),
    )
    geometry = pulse.Geometry(
        **load_geometry(msh_file=Path(datadir) / "cell.msh")._asdict(),
        microstructure=microstructure,
    )

    gamma = dolfin.Constant(0.0)

    matparams = pulse.HolzapfelOgden.default_parameters()
    material = pulse.HolzapfelOgden(
        active_model="active_strain",
        activation=gamma,
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    spring = 1.0
    robin_bc = [
        pulse.RobinBC(
            value=dolfin.Constant(spring), marker=geometry.markers["Left"][0]
        ),
        pulse.RobinBC(
            value=dolfin.Constant(spring), marker=geometry.markers["Right"][0]
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

    V_DG1 = dolfin.FunctionSpace(geometry.mesh, "DG", 1)
    proj = Projector(V_DG1)
    sigma_xx = dolfin.Function(V_DG1)
    sigma_r = dolfin.Function(V_DG1)
    sigma_c = dolfin.Function(V_DG1)
    E_xx = dolfin.Function(V_DG1)
    E_r = dolfin.Function(V_DG1)
    E_c = dolfin.Function(V_DG1)

    N = 50
    t = np.linspace(0, 1, N)
    amp = ca_transient(t)
    np.save(Path(resultsdir) / "t.npy", t)
    np.save(Path(resultsdir) / "gamma.npy", amp)

    output = Path(resultsdir) / "results.xdmf"
    output.unlink(missing_ok=True)
    output.with_suffix(".h5").unlink(missing_ok=True)

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


def postprocess(resultsdir="results", datadir="data", figdir="figures"):
    output = Path(resultsdir) / "results.xdmf"
    geo = load_geometry(msh_file=Path(datadir) / "cell.msh")
    V_DG1 = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    V_CG1 = dolfin.FunctionSpace(geo.mesh, "CG", 1)

    sigma_xx = dolfin.Function(V_DG1)
    sigma_r = dolfin.Function(V_DG1)
    E_xx = dolfin.Function(V_DG1)
    E_r = dolfin.Function(V_DG1)
    p = dolfin.Function(V_CG1)

    with h5py.File(output.with_suffix(".h5"), "r") as f:
        N = len(f["u"].keys())

    points = np.arange(0, 1.1, 0.1)

    sigma_xx_arr = np.zeros((N, len(points)))
    sigma_r_arr = np.zeros((N, len(points)))
    E_xx_arr = np.zeros((N, len(points)))
    E_r_arr = np.zeros((N, len(points)))
    p_arr = np.zeros((N, len(points)))

    with dolfin.XDMFFile(output.as_posix()) as f:
        for i in range(N):
            f.read_checkpoint(sigma_xx, "sigma_xx", i)
            f.read_checkpoint(sigma_r, "sigma_r", i)
            f.read_checkpoint(E_xx, "E_xx", i)
            f.read_checkpoint(E_r, "E_r", i)
            f.read_checkpoint(p, "p", i)

            for j, point in enumerate(points):
                sigma_xx_arr[i, j] = sigma_xx(0, 0, point)
                sigma_r_arr[i, j] = sigma_r(0, 0, point)
                E_xx_arr[i, j] = E_xx(0, 0, point)
                E_r_arr[i, j] = E_r(0, 0, point)
                E_r_arr[i, j] = E_r(0, 0, point)
                E_r_arr[i, j] = E_r(0, 0, point)
                p_arr[i, j] = p(0, 0, point)

    t = np.load(Path(resultsdir) / "t.npy")
    gamma = np.load(Path(resultsdir) / "gamma.npy")

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 8))

    lines = []
    labels = []
    for j, point in enumerate(points):
        ax[0, 0].plot(t, sigma_xx_arr[:, j], color=cm.tab20(point))
        ax[1, 0].plot(t, sigma_r_arr[:, j], color=cm.tab20(point))
        ax[2, 0].plot(t, p_arr[:, j], color=cm.tab20(point))
        ax[0, 1].plot(t, E_xx_arr[:, j], color=cm.tab20(point))
        (l,) = ax[1, 1].plot(t, E_r_arr[:, j], color=cm.tab20(point))
        lines.append(l)
        labels.append(f"$r = {point:.1f}$")

    ax[0, 0].set_title(r"$\sigma_{xx}$")
    ax[1, 0].set_title(r"$\sigma_{r}$")
    ax[2, 0].set_title("$p$")
    ax[0, 1].set_title(r"$E_{xx}$")
    ax[1, 1].set_title(r"$E_{r}$")
    ax[2, 1].plot(t, gamma)
    ax[2, 1].set_title(r"$\gamma$")
    fig.subplots_adjust(right=0.87)
    fig.legend(lines, labels, loc="center right")
    fig.savefig(Path(figdir) / "results.png")


if __name__ == "__main__":
    main()
    postprocess()
