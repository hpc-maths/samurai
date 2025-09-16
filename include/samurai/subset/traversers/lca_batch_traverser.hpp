// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include "../../level_cell_array.hpp"
#include "set_traverser_base.hpp"

#pragma once

namespace samurai
{
    template <LCA_concept LCA>
    class LCABatchTraverser;

    template <LCA_concept LCA>
    struct SetTraverserTraits<LCABatchTraverser<LCA>>
    {
        using interval_t         = typename LCA::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <LCA_concept LCA>
    class LCABatchTraverser : public SetTraverserBase<LCABatchTraverser<LCA>>
    {
        using Self = LCABatchTraverser<LCA>;
        using Base = SetTraverserBase<Self>;

      public:

        using interval_t               = typename Base::interval_t;
        using current_interval_t       = typename Base::current_interval_t;
        using value_t                  = typename Base::value_t;
        using vector_interval_iterator = typename std::vector<interval_t>::const_iterator;
        using list_interval_iterator   = typename std::vector<interval_t>::const_iterator;

        LCABatchTraverser(const vector_interval_iterator first, const vector_interval_iterator end)
            : m_vector_first_interval(first)
            , m_vector_end_interval(end)
            , m_use_vector_iterator(true)
        {
        }
        
        LCABatchTraverser(const list_interval_iterator first, const list_interval_iterator end)
            : m_list_first_interval(first)
            , m_list_end_interval(end)
            , m_use_vector_iterator(false)
        {
        }

        inline bool is_empty_impl() const
        {
            return m_use_vector_iterator ? m_vector_first_interval == m_vector_end_interval : m_list_first_interval == m_list_end_interval;
        }

        inline void next_interval_impl()
        {
            assert(!is_empty_impl());
            if (m_use_vector_iterator)
            {
				++m_vector_first_interval;
			}
			else
			{
				++m_list_first_interval;
			}
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_use_vector_iterator ? *m_vector_first_interval : *m_list_first_interval;
        }

      private:

        vector_interval_iterator m_vector_first_interval;
        vector_interval_iterator m_vector_end_interval;
        list_interval_iterator   m_list_first_interval;
        list_interval_iterator   m_list_end_interval;
        bool                     m_use_vector_iterator;
    };
}
