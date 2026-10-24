// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The schemes and their PETSc assemblies are stored by value in the solvers, whose
// move operations are noexcept only if moving them cannot throw. A user-declared
// destructor silently removes the implicit move operations and turns every move
// into a copy, so these checks guard against that regression at compile time.

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/block_operator.hpp>
#include <samurai/schemes/fv.hpp>

namespace samurai
{
    namespace
    {
        template <class T>
        constexpr bool is_nothrow_movable_v = std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>;

        constexpr std::size_t dim = 2;
        using Mesh                = MRMesh<mesh_config<dim>>;
        using scalar_field_t      = ScalarField<Mesh, double>;
        using vector_field_t      = VectorField<Mesh, double, dim>;

        // Linear homogeneous flux-based scheme
        using diffusion_t = decltype(make_diffusion_order2<vector_field_t>());
        // Linear heterogeneous flux-based scheme
        using convection_lin_t = decltype(make_convection_upwind<scalar_field_t>(std::declval<VelocityVector<dim>>()));
        // Non-linear flux-based scheme
        using convection_nonlin_t = decltype(make_convection_upwind<vector_field_t>());
        // Cell-based schemes
        using identity_t          = decltype(make_identity<scalar_field_t>());
        using cell_based_nonlin_t = decltype(make_diffusion_cell_based_nonlin<scalar_field_t>());
        // Sum of a cell-based and a flux-based scheme (implicit heat equation)
        using heat_t = decltype(std::declval<const identity_t&>() + std::declval<const decltype(make_diffusion_order2<scalar_field_t>())&>());
        // Block operator (Stokes)
        using stokes_t = decltype(make_block_operator<2, 2>(std::declval<const diffusion_t&>(),
                                                            std::declval<const decltype(make_gradient_order2<scalar_field_t>())&>(),
                                                            std::declval<const decltype(make_divergence_order2<vector_field_t>())&>(),
                                                            std::declval<const decltype(make_zero_operator<scalar_field_t>())&>()));

        static_assert(is_nothrow_movable_v<diffusion_t>);
        static_assert(is_nothrow_movable_v<convection_lin_t>);
        static_assert(is_nothrow_movable_v<convection_nonlin_t>);
        static_assert(is_nothrow_movable_v<identity_t>);
        static_assert(is_nothrow_movable_v<cell_based_nonlin_t>);
        static_assert(is_nothrow_movable_v<heat_t>);
        static_assert(is_nothrow_movable_v<stokes_t>);

        static_assert(is_nothrow_movable_v<Explicit<convection_lin_t>>);
        static_assert(is_nothrow_movable_v<Explicit<convection_nonlin_t>>);
        static_assert(is_nothrow_movable_v<Explicit<identity_t>>);
        static_assert(is_nothrow_movable_v<Explicit<heat_t>>);

#ifdef SAMURAI_WITH_PETSC
        template <class Scheme>
        using assembly_t = decltype(petsc::make_assembly(std::declval<const Scheme&>()));

        template <petsc::BlockAssemblyType type>
        using block_assembly_t = decltype(petsc::make_assembly<type>(std::declval<const stokes_t&>()));

        static_assert(is_nothrow_movable_v<assembly_t<diffusion_t>>);
        static_assert(is_nothrow_movable_v<assembly_t<convection_lin_t>>);
        static_assert(is_nothrow_movable_v<assembly_t<convection_nonlin_t>>);
        static_assert(is_nothrow_movable_v<assembly_t<identity_t>>);
        static_assert(is_nothrow_movable_v<assembly_t<cell_based_nonlin_t>>);
        static_assert(is_nothrow_movable_v<assembly_t<heat_t>>);
        static_assert(is_nothrow_movable_v<block_assembly_t<petsc::BlockAssemblyType::Monolithic>>);
        static_assert(is_nothrow_movable_v<block_assembly_t<petsc::BlockAssemblyType::NestedMatrices>>);

        // The numbering is owned by the assembly and referenced by its blocks or summands: a copy would share it.
        static_assert(!std::is_copy_constructible_v<assembly_t<diffusion_t>>);
        static_assert(!std::is_copy_assignable_v<assembly_t<diffusion_t>>);
        static_assert(!std::is_copy_constructible_v<block_assembly_t<petsc::BlockAssemblyType::Monolithic>>);
        static_assert(!std::is_copy_assignable_v<block_assembly_t<petsc::BlockAssemblyType::Monolithic>>);
#endif
    }

    TEST(scheme_move_semantics, moved_scheme_keeps_its_definition)
    {
        auto conv = make_convection_upwind<vector_field_t>();
        conv.set_name("conv");
        auto moved = std::move(conv);
        EXPECT_EQ(moved.name(), "conv");
        for (std::size_t d = 0; d < dim; ++d)
        {
            EXPECT_TRUE(static_cast<bool>(moved.flux_definition()[d].cons_flux_function));
        }
    }
}
