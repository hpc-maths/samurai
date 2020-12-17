Test cases LBM-MR
==========================================


D1Q2 scheme for the advection and the Burgers equations
-----------------------------

The target problem is the conservation law

.. math::
    \partial_{t} u + \partial_{x} (\varphi(u)) = 0, \qquad \text{with} \quad \begin{cases}
                                                                                \varphi(u) = 3/4u, \qquad &\text{advection}, \\
                                                                                \varphi(u) = u^2, \qquad &\text{Burgers}.
                                                                             \end{cases}





