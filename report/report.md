---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
title: Methods
description: Simple example of a contracting cylinder in FEniCS
keywords: ["cardiac mechancis", "elasticity", "stress"]
date: "2023-05-21"
license: CC-BY-4.0
authors: 
  - name: Henrik Finsberg
    affiliations:
      - Simula Research Laboratory
github: finsberg/contracting-cylinder
exports:
  - format: pdf
    template: curvenote
    output: report.pdf
  - format: tex
    template: curvenote
    output: report.tex
---
## Domain

We consider a cylindrical domain $\Omega(a, r) = [-a, a] \times \omega(r) $ with 

```{math}
\omega(r) = \{ (y, z) \in \mathbb{R}^2 : y^2 + z^2 < r \}
```

We let $a = 1000 \mu m$ and $r = 400 \mu m$ and will refer to $\Omega_0$ as $\Omega(400, 1000)$


```{figure} figures/domain.png
:name: cyl_domain
:alt: Cylindrical domain
:align: center

Example domain with $a = 1000 \mu m$ and $r = 400 \mu m$
```



## Material model

We use the the transversely isotropic version of the [Holzapfel Ogden model](https://doi.org/10.1098/rsta.2009.0091), i.e

```{math}
  \Psi(\mathbf{F})
  = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
  + \frac{a_f}{2 b_f} \mathcal{H}(I_{4\mathbf{f}_0} - 1)
  \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right)
```

with

```{math}
(x)_+ = \max\{x,0\}
```

and

```{math}
\mathcal{H}(x) = \begin{cases}
    1, & \text{if $x > 0$} \\
    0, & \text{if $x \leq 0$}
\end{cases}
```

is the Heaviside function. Here 

```{math}
I_1 = \mathrm{tr}(\mathbf{F}^T\mathbf{F})
```

and 

```{math}
I_{4\mathbf{f}_0} = (\mathbf{F} \mathbf{f}_0)^T \cdot \mathbf{F} \mathbf{f}_0
```

with $\mathbf{f}_0$ being the direction the muscle fibers. In our case we orient the fibers along the height the cylinder, i.e 

```{math}
\mathbf{f}_0 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}
```


```{figure} figures/fiber.png
:name: fibers
:alt: Fibers
:align: center

Showing the orientation of the cardiac fibers
```

### Material parameters

The material parameter are

| Parameter | Value [Pa]  |
|-----------|-------------|
| $a$       | 2280        |
| $b$       | 9.726       |
| $a_f$     | 1685        |
| $b_f$     | 15.779      |


## Modeling of active contraction

Similar to [Finsberg et. al](doi.org/10.1002/cnm.2982) we use an active strain formulation and decompose the deformation gradient into an active and an elastic part

```{math}
\mathbf{F} = \mathbf{F}_e \mathbf{F}_a
```

with 

```{math}
\mathbf{F}_a = (1 - \gamma) \mathbf{f} \otimes \mathbf{f}  + \frac{1}{\sqrt{1 - \gamma}} (\mathbf{I} - \mathbf{f} \otimes \mathbf{f}).
```

In these experiments we also set 

```{math}
\gamma(t) = \begin{cases}
\gamma_{\mathrm{min}} & \text{ if } t <= t_0, \\
\frac{\gamma_{\mathrm{max}} - \gamma_{\mathrm{min}}}{\beta} \left( e^{-\frac{t - t_0}{\tau_1}} - e^{-\frac{t - t_0}{\tau_2}} \right) + \gamma_{\mathrm{min}}  & \text{ otherwise} \\ 
\end{cases}
```

with

```{math}
\beta = \left(\frac{\tau_1}{\tau_2}\right)^{- \frac{1}{\frac{\tau_1}{\tau_2} - 1}} - \left(\frac{\tau_1}{\tau_2}\right)^{- \frac{1}{1 - \frac{\tau_2}{\tau_1}}}.
```

We choose the following parameters

| Parameter                | Value       |
|--------------------------|-------------|
| $t_0$                    | 0.05        |
| $\gamma_{\mathrm{min}}$  | 0.0         |
| $\gamma_{\mathrm{min}}$  | 0.2         |
| $\tau_1$                 | 0.05        |
| $\tau_2$                 | 0.11        |



```{code-cell} python
import numpy as np
import pandas as pd
import altair as alt


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

t = np.linspace(0, 1, 50)
amp = ca_transient(t, ca_ampl=0.2)

alt.Chart(pd.DataFrame({"t": t, "amp": amp})).mark_line()
```


## Variational formulation

We model the myocardium as incompressible using a two field variational approach and $\mathbb{P}_2-\mathbb{P}_1$ finite elements for the displacement $\mathbf{u}$ and hydrostatic pressure $p$.
m

The Euler‐Lagrange equations in the Lagrangian form reads: find $(\mathbf{u},p) \in H^1(\Omega_0) \times L^2(\Omega_0)$ such that for all $(\delta \mathbf{u},\delta \mathbf{u}) \in H^1(\Omega_0) \times L^2(\Omega_0)$ we have

```{math}
\delta \Pi(\mathbf{u}, p) = \int_{\Omega_0}\left[ \mathbf{P} : \nabla \delta \mathbf{u} - \delta p (J - 1) - pJ \mathbf{F}^{-T}: \delta \mathbf{u} \right] \mathrm{d}V + \int_{\partial \Omega_0^{a} \cup \Omega_0^{-a}} k \mathbf{u} \cdot \delta \mathbf{u}  \mathrm{d}S
```

where $ \Omega_0^{a} = \{ a \} \times \omega(r) $ represents the boundaries at each end of the cylinder. Here we enforce a Robin type boundary condition at both ends of the cylinder with a spring $k$. By default we choose $k=$ 0.1 Pa / $\mu$m$^2$



## Cauchy stress

The Cauchy stress tensor is given by

```{math}
\sigma = \mathbf{F} \frac{\partial \Psi }{\partial \mathbf{F}}= a e^{b (I_1 - 3)} \mathbf{B} - p \mathbf{I} + 2 a_f (I_{4\mathbf{f}_0} - 1) e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} \mathbf{f} \otimes \mathbf{f}
```

and has same units as the material parameters. Here $\mathbf{B} = \mathbf{F}\mathbf{F}^T$.
Note that we 


### Stress components

We can extract different components of the Cauchy stress tensor. The fiber component is oriented in along the $x$-direction and we can extract the cauchy stress along this direction using the following formula

```{math}
\sigma_{xx} = \mathbf{f}_0^T \cdot \sigma \mathbf{f}_0
```

In a cylinder there is a natural decomposition into radial and circumferential components. We have the radial basis function given by

```{math}
\mathbf{e}_r = \begin{pmatrix}
0 \\ y / (y^2 + z^2) \\ z / (y^2 + z^2)
\end{pmatrix}
```

```{figure} figures/rad.png 
:name: radial
:alt: Radial
:align: center

Showing the orientation of radial components
```

and the circumferential basis function given by

```{math}
\mathbf{e}_r = \begin{pmatrix}
0 \\ z / (y^2 + z^2) \\ -y / (y^2 + z^2)
\end{pmatrix}
```

```{figure} figures/circ.png 
:name: circ
:alt: Circ
:align: center

Showing the orientation of circular components
```