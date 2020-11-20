TP Burgers
==========================================


We want to approximate the solution :math:`u: [0, T] \times \mathbb{R} \to \mathbb{R}` of the well-known 1D Burgers equation

.. math::
    \begin{cases}
        \partial_t u(t, x) + \partial_x ( \varphi(u)(t, x) ) = 0, \qquad t \in [0, T], \quad &x \in \mathbb{R}, \\
        u(t=0, x) = u_0(x), \qquad  &x \in \mathbb{R},
    \end{cases}

with flux :math:`\varphi (u) = u^2/2` and hat-like initial datum given by


.. math::
    u_0(x) = (1+x) \chi_{[-1, 0]}(x) + (1-x) \chi_{[0, 1]}(x).


Using the method of the characteristic with the Rankine-Hugoniot jump relation, it can be shown that the solution is given by

.. math::
    u(t, x) = \frac{1+x}{1+t} \chi_{[-1, t]}(x) + \frac{1-x}{1-t} \chi_{[t, 1]}(x),


so that we can say that the solution blows up at time :math:`T^{\star}` in the sense that

.. math::
    \begin{cases}
        u(t, \cdot) \in C^0 (\mathbb{R}) \cap &L^{\infty}(\mathbb{R}), \qquad t \in [0, T^{\star}), \\
        u(t, \cdot) \in &L^{\infty}(\mathbb{R}), \qquad t \in [T^{\star}, T]. \\
    \end{cases}


We consider to work on a bounded domain, with cells of size :math:`\Delta x > 0` given by  ....


We consider that 

.. math::
    \overline{u}_{k}^n \simeq \frac{1}{\Delta x} \int_{x_k - \Delta x/2}^{x_k + \Delta x/2} u(t^n, x) \text{d}x.

The numerical Finite Volumes scheme is comes under the form

.. math::
    \overline{u}^{n+1}_k = \overline{u}^{n}_k + \frac{\Delta t}{\Delta x} (F_{k - 1/2}^n - F_{k+1/2}^n), 

where we utilize the upwind fluxes given by

.. math::
    F_{k - 1/2}^n = \mathcal{F}(\overline{u}^{n}_{k-1}, \overline{u}^{n}_k), \qquad \text{with} \quad 
     \mathcal{F}(\overline{u}_L, \overline{u}_R) = \begin{cases}
                                                        \varphi(\overline{u}_L), \qquad \text{if} \quad \frac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} &\geq 0, \\
                                                        \varphi(\overline{u}_R), \qquad \text{if} \quad \frac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} &< 0.
                                                  \end{cases}

Another possible choice for the flux is given by the Lax-Friedrichs, which is generally more diffusive than the upwind flux

.. math::
    \mathcal{F}(\overline{u}_L, \overline{u}_R) = \frac{1}{2} (\varphi(\overline{u}_L) + \varphi(\overline{u}_R)) - \frac{\Delta t}{2\Delta x} (\overline{u}_R - \overline{u}_L).


To perform the AMR adaptation, we employ the following criterion

.. math::
    \text{Split }C_{j, k} \quad \text{if} \quad |\partial_x \overline{u}_{j, k}| > \delta,

where the derivative on the cell is estimated with the following centered formula

.. math::
    \partial_x \overline{u}_{j, k} \simeq \frac{\overline{u}_{j, k + 1} - \overline{u}_{j, k - 1}}{2\Delta x_j}



