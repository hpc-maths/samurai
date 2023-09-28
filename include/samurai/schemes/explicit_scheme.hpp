#pragma once

namespace samurai
{
    template <class Scheme, class check = void>
    class Explicit
    {
        template <class>
        static constexpr bool dependent_false = false;

        static_assert(dependent_false<typename Scheme::cfg_t>,
                      "Either the required file has not been included, or the Explicit class has not been specialized for this type of scheme.");
    };

    template <class Scheme>
    auto make_explicit(const Scheme& s)
    {
        return Explicit<std::decay_t<Scheme>>(s);
    }
} // end namespace samurai
