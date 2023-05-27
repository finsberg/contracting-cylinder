from pathlib import Path
import typing
import json

import dolfin
import meshio


class MarkerFunctions(typing.NamedTuple):
    vfun: dolfin.MeshFunction | None = None
    efun: dolfin.MeshFunction | None = None
    ffun: dolfin.MeshFunction | None = None
    cfun: dolfin.MeshFunction | None = None


class Geometry(typing.NamedTuple):
    mesh: dolfin.Mesh
    markers: dict[str, tuple[int, int]]
    marker_functions: MarkerFunctions


def create_mesh(mesh, cell_type):
    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    if cells.size == 0:
        return None

    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    return out_mesh


def read_meshfunction(fname, obj):
    try:
        with dolfin.XDMFFile(Path(fname).as_posix()) as f:
            f.read(obj, "name_to_read")
    except RuntimeError:
        pass


def load_geometry(
    msh_file: Path | str,
    outdir: Path | str | None = None,
    unlink: bool = False,
) -> Geometry:
    if outdir is None:
        outdir = Path(msh_file).absolute().parent
    else:
        outdir = Path(outdir)

    triangle_mesh_name = outdir / "triangle_mesh.xdmf"
    tetra_mesh_name = outdir / "mesh.xdmf"
    marker_name = outdir / "markers.json"

    if dolfin.MPI.comm_world.size == 1:
        # Only use gmsh when running in serial
        msh = meshio.gmsh.read(msh_file)

        outdir.mkdir(exist_ok=True, parents=True)
        triangle_mesh = create_mesh(msh, "triangle")
        tetra_mesh = create_mesh(msh, "tetra")

        if triangle_mesh is not None:
            meshio.write(triangle_mesh_name, triangle_mesh)

        if tetra_mesh is None:
            raise RuntimeError("Unable to create mesh")

        meshio.write(
            tetra_mesh_name,
            tetra_mesh,
        )
        markers = {k: [int(vi) for vi in v] for k, v in msh.field_data.items()}
        marker_name.write_text(json.dumps(markers))

    markers = json.loads(marker_name.read_text())
    mesh = dolfin.Mesh()

    with dolfin.XDMFFile(tetra_mesh_name.as_posix()) as infile:
        infile.read(mesh)

    ffun_val = dolfin.MeshValueCollection("size_t", mesh, 2)
    read_meshfunction(triangle_mesh_name, ffun_val)
    ffun = dolfin.MeshFunction("size_t", mesh, ffun_val)
    ffun.array()[ffun.array() == max(ffun.array())] = 0
    if unlink:
        triangle_mesh_name.unlink(missing_ok=True)
        triangle_mesh_name.with_suffix(".h5").unlink(missing_ok=True)
    else:
        ffun_path = outdir / "ffun.xdmf"
        with dolfin.XDMFFile(ffun_path.as_posix()) as infile:
            infile.write(ffun)

    return Geometry(
        mesh=mesh, markers=markers, marker_functions=MarkerFunctions(ffun=ffun)
    )
