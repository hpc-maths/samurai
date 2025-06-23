// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"

namespace samurai
{

    template <class Set>
    struct SetTraits<Projection<Set>>
    {
        using interval_t = typename SetTraits<Set>::interval_t;

        static constexpr std::size_t dim = SetTraits<Set>::dim;
    };

    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
      public:

        Projection(const Set& set, const std::size_t level)
            : m_set(set)
            , m_level(level)
            , m_min_level(std::min(set.min_level(), level))
            , m_ref_level(std::max(set.ref_level(), level))
        {
        }

        std::size_t min_level() const
        {
            return m_min_level;
        }

        std::size_t level() const
        {
            return m_level;
        }

        std::size_t ref_level() const
        {
            return m_ref_level;
        }

        bool exists() const
        {
            return m_set.exists();
        }

        bool empty() const
        {
            return m_set.empty();
        }

      private:

        Set m_set;
        std::size_t m_level;
        std::size_t m_min_level;
        std::size_t m_ref_level;
    };

} // namespace samurai
