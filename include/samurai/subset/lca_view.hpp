// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/lca_traverser.hpp"

namespace samurai
{

	template <class LCA>
    class LCAView;

    template <class LCA>
    struct SetTraits<LCAView<LCA>>
    {
		static_assert(std::same_as<LevelCellArray<LCA::dim, typename LCA::interval_t>, LCA>);
		
        template <std::size_t>
        using traverser_t = LCATraverser<LCA>;
        
        static constexpr std::size_t dim = LCA::dim;
    };

    template <class LCA>
    class LCAView : public SetBase<LCAView<LCA>>
    {
        using Self = LCAView<LCA>;
    public:
		SAMURAI_SET_TYPEDEFS
		SAMURAI_SET_CONSTEXPRS
		
		LCAView(const LCA& lca) : m_lca(lca) {}
		
	    inline std::size_t level_impl() const
        {
			return m_lca.level();
        }

        inline bool exist_impl() const
        {
			return !empty_impl();
        }

        inline bool empty_impl() const
        {
			return m_lca.empty();
        }
        
        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d>) const
        {
			if constexpr (d != dim - 1)
            {
                const auto& y           = index[d];
                const auto& y_intervals = m_lca[d + 1];
                const auto& y_offsets   = m_lca.offsets(d + 1);
                // we need to find an interval that contains y.
                const auto y_interval_it = std::find_if(y_intervals.cbegin(),
                                                        y_intervals.cend(),
                                                        [y](const auto& y_interval)
                                                        {
                                                            return y_interval.contains(y);
                                                        });
                if (y_interval_it != y_intervals.cend())
                {
                    const std::size_t y_offset_idx = std::size_t(y + y_interval_it->index);
                    
                    return traverser_t<d>(m_lca[d].cbegin() + ptrdiff_t(y_offsets[y_offset_idx]),
                                          m_lca[d].cbegin() + ptrdiff_t(y_offsets[y_offset_idx + 1]));
                }
                else
                {
                    return traverser_t<d>(m_lca[d].cend(), m_lca[d].cend());
                }
            }
            else
            {
                return traverser_t<d>(m_lca[d].cbegin(), m_lca[d].cend());
            }
        }
    
    private:
		const LCA& m_lca;
	};
	
	
	template<std::size_t Dim, class TInterval> 
	LCAView<LevelCellArray<Dim, TInterval>> self(const LevelCellArray<Dim, TInterval>& lca)
	{
		return LCAView<LevelCellArray<Dim, TInterval>>(lca);
	}
	
} // namespace samurai
