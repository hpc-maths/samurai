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

    enum class Get
    {
        Cells,
        Intervals
    };

    template <class cfg>
    using StencilCells = CollapsStdArray<typename cfg::input_field_t::cell_t, cfg::stencil_size>;

    template <class cfg>
    using JacobianMatrix = CollapsMatrix<typename cfg::input_field_t::value_type, cfg::output_field_size, cfg::input_field_t::size>;

    template <class cfg>
    using StencilJacobian = StdArrayWrapper<JacobianMatrix<cfg>, cfg::stencil_size>;

} // end namespace samurai
