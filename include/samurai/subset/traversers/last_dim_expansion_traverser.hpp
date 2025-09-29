// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

	template<class SetTraverser>
	class LastDimExpansionTraverser;

    template <class SetTraverser>
    struct SetTraverserTraits<LastDimExpansionTraverser<SetTraverser>>
    {
        static_assert(IsSetTraverser<SetTraverser>::value);

        using interval_t         = typename SetTraverserTraits<SetTraverser>::interval_t;
        using current_interval_t = const interval_t&;
    };
    
    template<class SetTraverser>
	class LastDimExpansionTraverser : public SetTraverserBase< LastDimExpansionTraverser<SetTraverser> >
	{
		using Self = LastDimExpansionTraverser<SetTraverser>;
	public:
		SAMURAI_SET_TRAVERSER_TYPEDEFS
		
		LastDimExpansionTraverser(const SetTraverser& set_traverser, const value_t expansion)
			: m_set_traverser(set_traverser)
			, m_expansion(expansion)
		{
			assert(m_expansion > 0);
			next_interval_impl();
		}
		
		inline bool is_empty_impl() const
		{
			return m_isEmpty;
		}
		
		inline void next_interval_impl()
		{
			m_isEmpty = m_set_traverser.is_empty();
			
			if (!m_isEmpty)
			{
				m_current_interval.start = m_set_traverser.current_interval().start - m_expansion;
				m_current_interval.end   = m_set_traverser.current_interval().end   + m_expansion;
				
				m_set_traverser.next_interval();
				while (!m_set_traverser.is_empty() and m_set_traverser.current_interval().start - m_expansion <= m_current_interval.end)
				{
					m_current_interval.end   = m_set_traverser.current_interval().end   + m_expansion;
					m_set_traverser.next_interval();
				}
			}
		}
		
		inline current_interval_t current_interval_impl() const
		{
			return m_current_interval;
		}
		
	private:
		SetTraverser m_set_traverser;
		value_t      m_expansion;
		interval_t   m_current_interval;
		bool         m_isEmpty;
	};

} // namespace samurai
