# Contracting cylinder

This repo is for testing cardiac mechanics model on a contracting cylinder.

## Install dependencies
To run the code in this repo you need FEniCS and gmsh with OpenCascade installed. The easiest way to do this, is to use the [following docker image](https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh).
Next, you need to install the requirements, 

```
python3 -m pip install -r requirements.txt
```

## Running
When executing

```
python3 main.py
```
you will first create a cylindrical mesh using `gmsh` and then run a twitch using a synthetic twitch curve. 
We will use the transversely isotropic Holzapfel Ogden material model and the active strain formulation. The model uses and two field incompressible formulation with P2P1 (Taylor Hood) elements. The cylinder contracts against a Robin (sprint type) boundary condition at both ends.


## Author
Henrik Finsberg


## License
MIT