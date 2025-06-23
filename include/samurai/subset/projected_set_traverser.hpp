// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <SetTraverser_concept SetTraverser>
    class ProjectedSetTraverser;

    template <SetTraverser_concept SetTraverser>
    struct SetTraverserTraits
    {
        using interval_t = typename SetTraverser::interval_t;

        static constexpr std::size_t dim = SetTraverser::dim;
    };

    // The general strategy here is that we have an inner SetTraverser that is allays forward to our current interval.
    // This allows to easily merge overlapping intervals
    template <SetTraverser_concept SetTraverser>
    class ProjectedSetTraverser : public SetTraverserBase<ProjectedSetTraverser<SetTraverser>>
    {
        using Self       = ProjectedSetTraverser<SetTraverser>;
        using Base       = SetTraverserBase<Self>;
        using interval_t = typename SetTraverserTraits<Self>::interval_t;
        using value_t    = typename interval_t::value_t;

        ProjectedSetTraverser(const SetTraverser& set_traverser)
            : m_set_traverser(set_traverser)
        {
            if (!m_set_traverser.empty())
            {
                m_current_interval.start = start();
                m_current_interval.end   = end();

                m_set_traverser.next_interval();

                while (!m_set_traverser.is_empty() && start() < m_current_interval.end)
                {
                    m_current_interval.end = end();
                    m_set_traverser.next_interval();
                }
                Base::init_current();
            }
            else
            {
                m_is_empty = true;
            }
            Base::init_current();
        }

        inline bool is_empty() const
        {
            return m_is_empty;
        }

        inline void next_interval()
        {
            if (!m_set_traverser.empty())
            {
                m_current_interval.start = start();
                m_current_interval.end   = end();

                m_set_traverser.next_interval();

                while (!m_set_traverser.is_empty() && start() < m_current_interval.end)
                {
                    m_current_interval.end = end();
                    m_set_traverser.next_interval();
                }
            }
            else
            {
                m_is_empty = true;
            }
        }

        inline interval_t& current_interval()
        {
            return m_current_interval;
        }

      private:

        value_t start() const
        {
            return (m_set_traverser.current_interval().start() >> m_shift2min) << m_shift2ref;
        }

        value_t end() const
        {
            return (m_set_traverser.current_interval().end() >> m_shift2min) << m_shift2ref;
        }

        interval_t m_current_interval;
        SetTraverser m_set_traverser;
        bool m_is_empty;
    };

}
