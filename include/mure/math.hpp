#pragma once

#include <type_traits>

namespace mure
{

    /// Constexpr version of power function with integral exponent.
    template <typename Base, typename Exp>
    constexpr inline
    Base ipow(Base base, Exp exp)
    {
        static_assert(std::is_integral<typename std::decay<Exp>::type>::value, "Exponent must be of integral type");
        return exp == 0 ? Base(1) : (base * ipow(base, exp-1));
    }

} // namespace mure
