=============
Local schemes
=============

The local schemes are part of the Finite Volume module, enabled by

.. code-block:: c++

    #include <samurai/schemes/fv.hpp>
    #include <samurai/petsc.hpp> // optional, necessary for implicit schemes

They are characterized by a function that applies to a field, whose computation only involves information located in the current mesh cell.
The C++ interface is built similarly to the :doc:`flux-based Finite Volume schemes <finite_volume_schemes>`.

Implementing local scheme
-------------------------

Consider the implementation of a local operator :math:`\mathcal{A}` that applies to a field :math:`u`.
Let :math:`v` denote the resulting field:

.. math::
    v = \mathcal{A}(u).

First of all, the structural information about the scheme must be declared.
It contains:

- the :code:`input_field_type`: the C++ type of the field :math:`u`.
- the :code:`output_field_components`: the number of components of field of the resulting field :math:`v`.
- the :code:`scheme_type`, to be selected amongst the values of

.. code-block:: c++

    enum class SchemeType
    {
        NonLinear,
        LinearHeterogeneous,
        LinearHomogeneous
    };

This configuration must be declared in a :code:`LocalCellSchemeConfig` static structure.
Here is an example:

.. code-block:: c++

    auto u = samurai::make_field<...>("u", mesh);

    using cfg  = samurai::LocalCellSchemeConfig<
                        SchemeType::NonLinear,      // scheme_type
                        decltype(u)::nb_components, // output_field_components (here identical to the number of components of input field)
                        decltype(u)>;               // input_field_type

Secondly, we create the operator from the configuration :code:`cfg`:

.. code-block:: c++

    auto A = samurai::make_cell_based_scheme<cfg>();

    A.set_scheme_function(...);
    A.set_jacobian_function(...); // only A is non-linear

The signature of the scheme function actually depends on the :code:`SchemeType` declared in :code:`cfg` (see sections below).

Once the operator is created and defined, it can be used in an explicit context

.. code-block:: c++

    auto v = A(u);

or in an implicit context

.. code-block:: c++

    auto b = samurai::make_field<...>("b", mesh);
    samurai::petsc::solve(A, u, b); // solves the equation A(u) = b

Note that the :code:`solve` function involves a linear or a non-linear solver according to the :code:`SchemeType` declared in :code:`cfg`.


Non-linear operators
--------------------

The analytical formula of the operator is implemented as a lambda function.

.. code-block:: c++

    A.set_scheme_function([&](auto& cell, const auto& field)
    {
        // Local field value
        auto v = field[cell];

        // Use 'v' and captured parameters in your computation
        samurai::SchemeValue<cfg> result = ...;

        return result;
    });

The parameters of the function are

- :code:`cell`: the current local cell;
- :code:`field`: the input field, to which the operator applies. Its actual type is declared in :code:`cfg`.

The return type :code:`SchemeValue<cfg>` is a array-like structure of size :code:`output_field_components` (declared in :code:`cfg`).
It is based on the :code:`xtensor` library, so all :code:`xtensor` functions and accessors can be used.
The :math:`i`-th component can be accessed with :code:`result(i)`.

.. note::
    If :code:`output_field_components` is set to 1, :code:`SchemeValue<cfg>` reduces to a scalar type (typically :code:`double`).

If the operator is to be implicited, its jacobian function must also be defined.
If only explicit applications of the operator shall be used, then this step is optional.

.. code-block:: c++

    A.set_jacobian_function([&](auto& cell, const auto& field)
    {
        // Local field value
        auto v = field[cell];

        samurai::JacobianMatrix<cfg> jac = ...
        return jac;
    });

.. warning::
    The type :code:`JacobianMatrix<cfg>` is a matrix of size :code:`output_field_components x input_field_type`.
    However, if :code:`output_field_components = input_field_components = 1`, it reduces to a scalar type (typically :code:`double`).
