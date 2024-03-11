===========================
Reaction-diffusion equation
===========================

In this tutorial, we detail how to compute the solution of a reaction-diffusion equation using the finite volume module.
The complete code of this tutorial can be downloaded here: :download:`Nagumo <../../../demos/FiniteVolume/nagumo.cpp>`.
We write the abstract reaction-diffusion equation as

.. math::
        \partial_t u + \mathcal{D}(u) = \mathcal{R}(u),

where :math:`\mathcal{D}` is a diffusion operator and :math:`\mathcal{R}` is a reaction operator.
The specific problem we tackle in this example is the Nagumo equation (also called Fisher-KPP equation),
which models a travelling wave.
Let :math:`\Omega\subset\mathbb{R}^d` be the domain of study in a :math:`d`-dimensional space.
Let :math:`T\in\mathbb{R}_+` be the final time of the simulation.
The problem reads: find :math:`u\colon [0, T]\times\Omega \to \mathbb{R}` such that

.. math::
        \partial_t u - D\Delta u = k u^2(1-u),

where :math:`D` and :math:`k` are constant, positive, scalar values.
The solution :math:`u` is subject to homogeneous Neumann boundary conditions on :math:`\partial\Omega`
and to the initial condition :math:`u = u_0` at :math:`t=0`.
In this example, we have

.. math::

        \mathcal{D}(u) &= - D\Delta u, \\
        \mathcal{R}(u) &= k u^2(1-u).


Standard samurai initializations
--------------------------------

The mesh is created by the following code:

.. code-block:: c++

    // Space dimension
    static constexpr std::size_t dim = 1;

    // Domain: [-10, 10]^dim
    double left_box  = -10;
    double right_box = 10;

    // Mesh creation
    using Config  = samurai::MRConfig<dim>;
    using Box     = samurai::Box<double, dim>;
    using point_t = typename Box::point_t;

    std::size_t min_level = 0;
    std::size_t max_level = 4;

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

Here, we have used the mesh type :code:`MRMesh`, but any other type of mesh can be chosen.
Then, solution fields at two time steps (:math:`u_n` and :math:`u_{n+1}`) are declared, and boundary conditions are attached:

.. code-block:: c++

    auto u    = samurai::make_field<1>("u",    mesh);
    auto unp1 = samurai::make_field<1>("unp1", mesh);

    samurai::make_bc<samurai::Neumann<1>>(u,    0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);

Finally, we code the initial condition:

.. code-block:: c++

    double D = 1;  // diffusion coefficient
    double k = 10; // reaction coefficient

    double z0 = left_box / 5;    // wave initial position
    double c  = sqrt(k * D / 2); // wave velocity

    auto beta = [&](double z)
    {
        double e = exp(-sqrt(k / (2 * D)) * (z - z0));
        return e / (1 + e);
    };

    auto exact_solution = [&](double x, double t)
    {
        return beta(x - c * t);
    };

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = exact_solution(cell.center(0), 0);
                           });

Spatial discretization
----------------------

We now build the operators required by the equation, namely, the diffusion and the reaction operators.

Diffusion operator
++++++++++++++++++

The diffusion operator of order 2 is implemented in the :doc:`finite volume framework <../reference/finite_volume_schemes>` and is declared by

.. code-block:: c++

    auto diff = samurai::make_diffusion<decltype(u)>(D);

Remark that the field type the diffusion operator applies to is given as a static (template) parameter,
and the diffusion coefficient :code:`D` is passed as a dynamic parameter.

.. note::

    Beware of the sign! The diffusion operator corresponds to :math:`-\Delta`.
    Keep it in mind when applying :code:`diff`.
    The function :code:`samurai::make_laplacian<...>()` provides the operator :math:`\Delta` (without the minus sign).
    The operator :code:`diff` as constructed above is strictly equivalent to

    .. code-block:: c++

        auto diff = -D * samurai::make_laplacian<decltype(u)>();

Reaction operator
+++++++++++++++++

The reaction operator is a local scheme, which we build using the :doc:`dedicated framework <../reference/local_schemes>`.
We start by declaring a configuration object that holds the static properties of the operator.

.. code-block:: c++

    using cfg  = samurai::LocalCellSchemeConfig<samurai::SchemeType::NonLinear, 1, decltype(u)>;

Here,

- :code:`LocalCellSchemeConfig<...>` indicates that the scheme is *local*;
- :code:`SchemeType::NonLinear` indicates that the scheme is *non-linear*;
- :code:`1` indicates the *output field size* (here, a scalar field);
- :code:`decltype(u)` indicates the *input field*.

Second, we create the reaction operator from the configuration :code:`cfg`.

.. code-block:: c++

    auto react = samurai::make_cell_based_scheme<cfg>();

Then, we implement the analytical formula of the operator as a lambda function.

.. code-block:: c++

    react.scheme_function() = [&](auto& cell, const auto& field)
    {
        auto v = field[cell];
        return k * v * v * (1 - v);
    };

The parameters of the function are

- :code:`cell`: the current local cell;
- :code:`field`: the input field, to which the operator applies. Its actual type is declared in the :code:`cfg` object.

If the operator is to be implicited, its jacobian function must also be defined.
If only explicit applications of the operator shall be used, then this step is optional.

.. code-block:: c++

    react.jacobian_function() = [&](auto& cell, auto& field)
    {
        auto v = field[cell];
        return k * (2 * v * (1 - v) - v * v);
    };

Identity operator
+++++++++++++++++

In order to implement an implicit scheme, the identity operator must also be declared.

.. code-block:: c++

    auto id = samurai::make_identity<decltype(u)>();

In an implicit context, this operator will generate the identity matrix.

Time integration
----------------

We consider here the Euler scheme with timestep :math:`dt`.

Implicit diffusion, explicit reaction
+++++++++++++++++++++++++++++++++++++

The Euler scheme reads

.. math::
    u_{n+1} + dt\,\mathcal{D}(u_{n+1}) = u_n + dt\,\mathcal{R}(u_n) .

:math:`u_{n+1}` is then computed by solving the linear equation

.. math::
    (Id +dt\,\mathcal{D})u_{n+1} = u_n + dt\,\mathcal{R}(u_n) \qquad \text{where } Id \text{ is the identity operator}.

In the discrete setting, the corresponding linear system is solved by the following code:

.. code-block:: c++

    auto implicit_operator = id + dt * diff;
    auto rhs               = u + dt * react(u);
    samurai::petsc::solve(implicit_operator, unp1, rhs);

The first instruction creates a new operator from an algebraic expression involving already declared operators.
The result is an operator, which is a very light object.
Especially, this is not a matrix, nothing is assembled or computed at this point.
The second instruction computes the right-hand side of the system.
The result is a field, here allocated by the instruction itself.
In a practical code, this instruction would be placed within a time loop,
so you might want to allocate :code:`rhs` before the loop
in order to avoid repeated memory allocations/deallocations.
The last instruction actually performs the computations: it assembles a PETSc matrix and solves the linear system.
The default PETSc configuration is conserved.
In particular, the linear solver is defaulted to the GMRES method with an ILU preconditioner with a tolerance of 1e-5.
To configure it otherwise, PETSc command line arguments must be used.
For instance, add :code:`-ksp_type preonly -pc_type lu` to the command line to use the LU factorization.
To hardcode solver parameters, or to conserve the solver for further use, a solver object must be created.
Instead of using the stand-alone solving function :code:`samurai::petsc::solve(...)`, you can write

.. code-block:: c++

    auto solver = samurai::petsc::make_solver(implicit_operator);
    solver.set_unknown(unp1);
    solver.solve(rhs);

Implicit diffusion and reaction
+++++++++++++++++++++++++++++++

The Euler scheme reads

.. math::
    u_{n+1} + dt\,\mathcal{D}(u_{n+1}) - dt\,\mathcal{R}(u_{n+1}) = u_n.

:math:`u_{n+1}` is then computed by solving the non-linear equation

.. math::
    (Id +dt\,\mathcal{D} - dt\,\mathcal{R})(u_{n+1}) = u_n \qquad \text{where } Id \text{ is the identity operator}.

In the discrete setting, the corresponding non-linear system is solved by the following code:

.. code-block:: c++

    auto implicit_operator = id + dt * diff - dt * react;
    unp1 = u; // set initial guess for the Newton algorithm
    samurai::petsc::solve(implicit_operator, unp1, u);

The first instruction creates, from an algebraic expression of operators, the operator to be implicited.
As the resulting operator is non-linear, a non-linear solver such as a Newton method shall be used.
An initial point to start the algorithm is required.
Therefore, the solution field, here :code:`unp1`, must be explicitly initialized.
In order not to fall into a local minimum, it is advised to choose a point knowingly close to the actual minimizer.
Selecting the solution at the current time step is a classical practice, which we do here as an example.
Finally, the last instruction solves the non-linear system using PETSc.
Just like for linear systems, the solver can be configured using PETSc command line arguments such as :code:`-snes_type` or :code:`-snes_tol`,
and a solver object can be declared instead of the :code:`solve(...)` function.

Remark that the :code:`solve(...)` instruction is identical to the one used for the linear equation of the preceding paragraph.
Indeed, there is no need to indicate what type of solver must be used (linear or non-linear):
this is determined by the :code:`SchemeType` associated to the operator that is fed to the solving function.
Here, the :code:`react` operator is configured with :code:`SchemeType::NonLinear`, which is then transferred to :code:`implicit_operator`,
indicating that a non-linear solver must be used within the :code:`solve(...)` function.

Time loop
+++++++++

The time loop can be written straightforwardly:

.. code-block:: c++

    bool explicit_reaction = true; // or false

    double T = 1;     // final time
    double dt = 0.01; // time step
    double t = 0;     // current time
    while (t < T)
    {
        // Move to next timestep
        t += dt;

        // Apply scheme
        if (explicit_reaction)
        {
            auto implicit_operator = id + dt * diff;
            auto rhs               = u + dt * react(u);
            samurai::petsc::solve(implicit_operator, unp1, rhs);
        }
        else
        {
            auto implicit_operator = id + dt * diff - dt * react;
            unp1 = u;
            samurai::petsc::solve(implicit_operator, unp1, u);
        }

        // u <-- unp1
        std::swap(u.array(), unp1.array());
    }

Mesh adaptation can finally be added to the program.
Refer to complete code :download:`Nagumo <../../../demos/FiniteVolume/nagumo.cpp>`.
