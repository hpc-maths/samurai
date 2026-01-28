#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class VelocityField, class TemperatureField>
    auto make_buoyancy(double factor = 1.0)
    {
        static_assert(VelocityField::dim >= 2, "Buoyancy operator is not implemented in 1D.");

        using cfg = samurai::LocalCellSchemeConfig<samurai::SchemeType::LinearHomogeneous, VelocityField, TemperatureField>;

        auto buoyancy = samurai::make_cell_based_scheme<cfg>("Buoyancy");

        buoyancy.coefficients_func() = [factor](samurai::StencilCoeffs<cfg>& sc, double)
        {
            static constexpr std::size_t y = 1;
            sc.fill(0.);
            sc(y, 0) = -factor;
        };
        return buoyancy;
    }
}
