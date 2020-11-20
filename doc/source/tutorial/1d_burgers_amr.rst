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
    \overline{u}_{j}^n \simeq \frac{1}{\Delta x} \int_{x_j - \Delta x/2}^{x_j + \Delta x/2} u(t^n, x) \text{d}x.

The numerical Finite Volumes scheme is comes under the form

.. math::
    \overline{u}^{n+1}_j = \overline{u}^{n}_j + \frac{\Delta t}{\Delta x} (F_{j - 1/2}^n - F_{j+1/2}^n), 

where we utilize the upwind fluxes given by

.. math::
    F_{j - 1/2}^n = \mathcal{F}(\overline{u}^{n}_{j-1}, \overline{u}^{n}_j), \qquad \text{with} \quad 
     \mathcal{F}(\overline{u}_L, \overline{u}_R) = \begin{cases}
                                                        \varphi(\overline{u}_L), \qquad \text{if} \quad \frac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} &\geq 0, \\
                                                        \varphi(\overline{u}_R), \qquad \text{if} \quad \frac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} &< 0.
                                                  \end{cases}




