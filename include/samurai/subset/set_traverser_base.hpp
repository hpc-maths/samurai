// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <cstddef>

#include <utility>

#pragma once

namespace samurai
{
    template <class SetTraverser>
    struct SetTraverserTraits;

    template <class Derived>
    class SetTraverserBase;

    template <typename T>
    concept SetTraverser_concept = std::is_base_of<SetTraverserBase<T>, T>::value;

    template <class Derived>
    class SetTraverserBase
    {
      public:

        using interval_t = typename SetTraits<Set>::interval_t;
        using value_t    = typename interval_t::value_t;

        static constexpr static constexpr std::size_t dim = SetTraits<Set>::dim;

        const Derived& derived_cast() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived_cast()
        {
            return static_cast<Derived&>(*this);
        }

        inline bool is_empty() const
        {
            return derived_cast()->is_empty();
        }

        inline void next_interval()
        {
            derived_cast()->next_interval();
        }

        inline interval_t& current_interval()
        {
            derived_cast()->current_interval()
        }

        inline void next(auto scan)
        {
            assert(!empty());

            if (m_current == scan)
            {
                if (m_is_current_at_start)
                {
                    m_current = current_interval().end;
                }
                else
                {
                    next_interval();
                }
                m_is_current_at_start = !m_is_current_at_start;
            }
        }

        inline bool is_in(auto scan) const
        {
            return (!is_empty()) && !((scan < m_current) ^ (!m_is_start));
        }

        inline value_t min() const
        {
            return m_current;
        }

        inline auto& current()
        {
            return m_current;
        }

      protected:

        inline void init_current()
        {
            m_current             = current_interval().start;
            m_is_current_at_start = true;
        }

        value_t m_current          = std::numeric_limits<value_t>::min();
        bool m_is_current_at_start = true;
    };
}
