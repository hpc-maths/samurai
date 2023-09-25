#pragma once

namespace samurai
{
    template <class Scheme, class check = void>
    class Explicit
    {
    };

    template <class Scheme>
    auto make_explicit(const Scheme& s)
    {
        return Explicit<std::decay_t<Scheme>>(s);
    }
} // end namespace samurai
