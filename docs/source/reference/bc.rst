Boundary condition
==================

In any solution of a physical problem using partial differential equations, the computational domain is bounded, so it is necessary to impose boundary conditions. These may be periodic or of other types, such as Dirichlet, Neumann, Robin, ... In |project|, we offer the possibility of imposing periodic conditions or so-called "classical" conditions (Dirichlet or Neumann). But the implementation of boundary conditions is flexible enough to allow users to implement and use their own boundary conditions, as we'll see later.

The characteristics of a boundary condition in |project| are as follows

1. a value to be imposed: this may be a constant or a space-dependent function
2. a region in which to impose this value: this can be the whole boundary, a boundary for a given direction, or a region.

In what follows, we'll give a number of examples to help you understand how they work. Don't hesitate to have a look at the examples in the ``demos`` directory, which use what we're going to describe here in concrete cases.

Boundary with constant values
-----------------------------

We give here a first use of edge conditions in |project| where we put the same condition on all the boundaries of the domain and this condition is a constant.

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::make_bc<samurai::Dirichlet>(u, 0.);

The number of parameters in the ``make_bc`` function depends on the size of the field. Here, the field size is $1$ so we have just one value to define.

Let's take an example where the field size is different to :math:`1`:

.. code-block:: c++

    auto u = samurai::make_field<double, 3>("my_field", mesh);

    samurai::make_bc<samurai::Dirichlet>(u, 1., 2., 3.);

Since we have a field of size :math:`3`, we have to define three values: one for each component. We said here that we want :cpp:class:`Dirichlet` boundary condition on each boundary of the domain with constant values: :math:`1` for the first component of the field, :math:`2` for the second component of the field, ...

If we want to set another boundary condition, we just change the type :cpp:class:`samurai::Dirichlet` used in :cpp:func:`samurai::make_bc`. For example, if we want to apply :cpp:class:`Neumann` boundary condition, we just have to write

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::make_bc<samurai::Neumann>(u, 0.);

Boundary using a function
-------------------------

If we want to impose boundary conditions which depend on the boundary coordinates, we have to define a lambda function which returns the value on the given point. Here is an example

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    samurai::make_bc<Dirichlet>(
        u,
        [](const auto& direction, const auto& cell_in, const auto& coord)
        {
            return coord[0]*coord[0];
        }
    );

The lambda function return :math:`x^2` where :math:`x` is the first coordinate on the boundary.

The given parameters of the lambda function are as follows:

- `direction` is an array of integer of size `dim` which indicates how to go out from the `cell_in`.
- `cell_in` is of type :cpp:class:`samurai::Cell` and gives the characteristics of the cell which has a boundary face.
- `coord` is an array of double of size `dim`given the center of the boundary face.

.. note::
    The output of the lambda function must be convertible to a xtensor container with the shape equals to the number of field components.

Boundary along a direction
--------------------------

If we want to impose boundary conditions on a domain face, we can define a direction and use it to describe the boundary where we want to set the condition.

.. code-block:: c++

    auto u = samurai::make_field<double, 1>("my_field", mesh);

    const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
    samurai::make_bc<samurai::Dirichlet>(u, -1.)->on(left);

    const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
    samurai::make_bc<samurai::Dirichlet>(u, 1.)->on(right);


Define your own boundary
------------------------

It's possible to describe your own boundary condition as it is done for :cpp:class:`samurai::Dirichlet` and :cpp:class:`samurai::Neumann`. For that, you have to define a class which is based on :cpp:class:`samurai::Bc` and which defines a method called `apply` explaining how to impose the boundary condition by populate a ghost cell. Let's take the example of a Dirichlet condition to better understand how it works.

.. code-block:: c++

    template <class Field>
    struct Dirichlet : public Bc<Field>
    {
        INIT_BC(Dirichlet)

        void apply(Field& f, const cell_t& cell_out, const cell_t& cell_in,
                   const value_t& value) const override
        {
            f[cell_out] = 2 * value - f[cell_in];
        }
    };

`INIT_BC` is a macro which defines some useful types and a clone method. The most important is the definition af the `apply` method where the parameters are fixed ans as follows:

- `f` is the field where the boundary conditions are applied (where the ghosts will be updated).
- `cell_out` is of type :cpp:class:`samurai::Cell` and gives the characteristics of the cell which is outside the boundary face.
- `cell_in` is of type :cpp:class:`samurai::Cell` and gives the characteristics of the cell which is inside the boundary face.
- `value` is an array of the type of the components of the field and with the size of the number of components.
