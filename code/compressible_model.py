import pulse
import dolfin


class CompressibleProblem(pulse.MechanicsProblem):
    """
    This class implements a compressbile model with a penalized
    compressibility term, solving for the displacement only.

    """

    def __init__(self, *args, kappa=1e3, **kwargs):
        self.kappa = dolfin.Constant(kappa)
        super().__init__(*args, *kwargs)

    def _init_spaces(self):
        mesh = self.geometry.mesh

        element = dolfin.VectorElement("P", mesh.ufl_cell(), 2)
        self.state_space = dolfin.FunctionSpace(mesh, element)
        self.state = dolfin.Function(self.state_space)
        self.state_test = dolfin.TestFunction(self.state_space)

        # Add penalty factor

    def _init_forms(self):
        u = self.state
        v = self.state_test

        F = dolfin.variable(pulse.kinematics.DeformationGradient(u))
        J = pulse.kinematics.Jacobian(F)

        dx = self.geometry.dx

        # Add penalty term
        internal_energy = self.material.strain_energy(F) + self.kappa * (
            J * dolfin.ln(J) - J + 1
        )

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )

        self._virtual_work += self._external_work(u, v)

        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )

        self._set_dirichlet_bc()
        self._init_solver()
