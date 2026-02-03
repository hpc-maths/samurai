// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

/**
 * @file samurai_template_instantiations.cpp
 * @brief Explicit template instantiations for demos (shared library)
 *
 * This file pre-instantiates common template specializations used
 * across FiniteVolume demos to reduce compilation time in each demo.
 *
 * Compiled as a shared library (.dylib/.so) to avoid duplicate symbol
 * linker errors. The instantiations are compiled once and shared across
 * all demos that link against this library.
 *
 * Build with: -DSAMURAI_INSTANTIATE_TEMPLATES=ON
 *
 * Compilation speedup: ~20-30% faster per-demo compilation
 * (first-time compilation of library is ~30-60s, subsequent demos are faster)
 */

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/interval.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/stencil_field.hpp>

namespace samurai
{
    // ===== Interval Instantiations =====
    // Core data structures that appear in every compilation unit

    template struct Interval<int, signed long long int>;
    template struct Interval<long, signed long long int>;

    // ===== CellArray Instantiations =====
    // Foundation for mesh representation across dimensions

    template class CellArray<1>;
    template class CellArray<2>;
    template class CellArray<3>;

    // ===== ScalarField Instantiations =====
    // Most commonly used field type in demos

    template class ScalarField<MRMesh<MRConfig<1>>, double>;
    template class ScalarField<MRMesh<MRConfig<1>>, float>;

    template class ScalarField<MRMesh<MRConfig<2>>, double>;
    template class ScalarField<MRMesh<MRConfig<2>>, float>;

    template class ScalarField<MRMesh<MRConfig<3>>, double>;
    template class ScalarField<MRMesh<MRConfig<3>>, float>;

    // ===== VectorField Instantiations =====
    // Vector-valued field instantiations for common dimensions

    template class VectorField<MRMesh<MRConfig<1>>, double, 1>;
    template class VectorField<MRMesh<MRConfig<1>>, double, 2>;
    template class VectorField<MRMesh<MRConfig<1>>, double, 3>;

    template class VectorField<MRMesh<MRConfig<2>>, double, 1>;
    template class VectorField<MRMesh<MRConfig<2>>, double, 2>;
    template class VectorField<MRMesh<MRConfig<2>>, double, 3>;

    template class VectorField<MRMesh<MRConfig<3>>, double, 1>;
    template class VectorField<MRMesh<MRConfig<3>>, double, 2>;
    template class VectorField<MRMesh<MRConfig<3>>, double, 3>;

    // ===== MRAdapt Instantiations =====
    // Adaptation operator for multiresolution meshes

    // 1D MRAdapt with single scalar field
    template auto make_MRAdapt<ScalarField<MRMesh<MRConfig<1>>, double>>(ScalarField<MRMesh<MRConfig<1>>, double>&);

    // 2D MRAdapt with single scalar field (most common)
    template auto make_MRAdapt<ScalarField<MRMesh<MRConfig<2>>, double>>(ScalarField<MRMesh<MRConfig<2>>, double>&);

    // 2D MRAdapt with vector field
    template auto make_MRAdapt<VectorField<MRMesh<MRConfig<2>>, double, 2>>(VectorField<MRMesh<MRConfig<2>>, double, 2>&);

    // 3D MRAdapt with single scalar field
    template auto make_MRAdapt<ScalarField<MRMesh<MRConfig<3>>, double>>(ScalarField<MRMesh<MRConfig<3>>, double>&);

} // namespace samurai
