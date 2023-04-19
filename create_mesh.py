import gmsh
import numpy as np


def main():
    gmsh.initialize()
    gmsh.model.add("Cell")
    char_length = 0.2
    L = 5
    r = 1
    cylinder = gmsh.model.occ.addCylinder(-L / 2, 0.0, 0.0, L, 0, 0, r)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.occ.getEntities(dim=2)
    # inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 3, 5, 7
    right = 1
    left = 2
    sides = 3

    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

        if np.allclose(com, [L / 2, 0, 0]):
            # Right boundary
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], right)
            gmsh.model.setPhysicalName(surface[0], surface[1], "Right")
        elif np.allclose(com, [-L / 2, 0, 0]):
            # Left boundary
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], left)
            gmsh.model.setPhysicalName(surface[0], surface[1], "Left")
        else:
            # Sides
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], sides)
            gmsh.model.setPhysicalName(surface[0], surface[1], "Sides")

    gmsh.model.addPhysicalGroup(3, [cylinder], sides)
    gmsh.model.setPhysicalName(3, cylinder, "Volume")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write("cell.msh")
    gmsh.finalize()


if __name__ == "__main__":
    main()