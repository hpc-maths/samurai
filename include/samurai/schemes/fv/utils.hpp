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
    using StencilCells = CollapsStdArray<typename cfg::input_field_t::cell_t, cfg::stencil_size>;

    template <class cfg>
    using StencilValues = CollapsStdArray<typename cfg::input_field_t::local_data_type, cfg::stencil_size>;

    template <class cfg>
    using JacobianMatrix = CollapsMatrix<typename cfg::input_field_t::value_type, cfg::output_n_comp, cfg::input_field_t::n_comp>;

    template <class cfg>
    using StencilJacobian = StdArrayWrapper<JacobianMatrix<cfg>, cfg::stencil_size>;

} // end namespace samurai
