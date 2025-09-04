#pragma once

#include "../../storage/containers.hpp"

namespace samurai
{
    enum class SchemeType
    {
        NonLinear,
        LinearHeterogeneous,
        LinearHomogeneous
    };

    template <class cfg>
    using StencilCells = CollapsStdArray<typename cfg::input_field_t::cell_t, cfg::stencil_size, true>;

    template <class cfg>
    using StencilValues = CollapsStdArray<typename cfg::input_field_t::local_data_type, cfg::stencil_size, cfg::input_field_t::is_scalar>;

    template <class cfg>
    using JacobianMatrix = CollapsMatrix<typename cfg::input_field_t::value_type,
                                         cfg::output_field_t::n_comp,
                                         cfg::input_field_t::n_comp,
                                         cfg::input_field_t::is_scalar && cfg::output_field_t::is_scalar>;

    template <class cfg>
    using StencilJacobian = collapsable_algebraic_std_array<JacobianMatrix<cfg>, cfg::stencil_size, true>;

} // end namespace samurai
