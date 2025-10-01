// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/expansion_traverser.hpp"
#include "traversers/last_dim_expansion_traverser.hpp"

namespace samurai
{

    template <class Set>
    class Expansion;

    template <class Set>
    struct SetTraits<Expansion<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == Set::dim - 1,
                                               LastDimExpansionTraverser<typename Set::template traverser_t<d>>,
                                               ExpansionTraverser<typename Set::template traverser_t<d>>>;

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    namespace detail
    {
        template <class Set, typename Seq>
        struct ExpansionWork;

        template <class Set, std::size_t... ds>
        struct ExpansionWork<Set, std::index_sequence<ds...>>
        {
            template <std::size_t d>
            using child_traverser_t = typename Set::template traverser_t<d>;

            template <std::size_t d>
            using array_of_child_traverser_offset_range_t = std::vector<typename MemoryPool<child_traverser_t<d>>::OffsetRange>;

            using Type = std::tuple<array_of_child_traverser_offset_range_t<ds>...>;

            static_assert(std::tuple_size<Type>::value == Set::dim - 1);
        };

    } // namespace detail

    template <class Set>
    class Expansion : public SetBase<Expansion<Set>>
    {
        using Self            = Expansion<Set>;
        using OffsetRangeWork = detail::ExpansionWork<Set, std::make_index_sequence<Set::dim - 1>>::Type;

      public:

        SAMURAI_SET_TYPEDEFS
        //~ SAMURAI_SET_CONSTEXPRS

        using expansion_t    = std::array<value_t, Base::dim>;
        using do_expansion_t = std::array<bool, Base::dim>;

        Expansion(const Set& set, const expansion_t& expansions)
            : m_set(set)
            , m_expansions(expansions)
        {
        }

        Expansion(const Set& set, const value_t expansion)
            : m_set(set)
        {
            m_expansions.fill(expansion);
        }

        Expansion(const Set& set, const value_t expansion, const do_expansion_t& do_expansion)
            : m_set(set)
        {
            for (std::size_t i = 0; i != m_expansions.size(); ++i)
            {
                m_expansions[i] = expansion * do_expansion[i];
            }
        }

        // we need to define a custom copy and move constructor because
        // we do not want to copy m_work_offsetRanges
        Expansion(const Expansion& other)
            : m_set(other.m_set)
            , m_expansions(other.m_expansions)
        {
        }

        Expansion(Expansion&& other)
            : m_set(other.m_set)
            , m_expansions(other.m_expansions)
        {
        }

        ~Expansion()
        {
            static_for<0, Base::dim - 1>::apply(
                [this](const auto d)
                {
                    using Work = MemoryPool<typename Set::template traverser_t<d>>;

                    auto& work = Work::getInstance();

                    for (auto& offset_range : std::get<d>(m_work_offsetRanges))
                    {
                        work.freeChunk(offset_range);
                    }
                });
        }

        inline std::size_t level_impl() const
        {
            return m_set.level();
        }

        inline bool exist_impl() const
        {
            return m_set.exist();
        }

        inline bool empty_impl() const
        {
            return m_set.empty();
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            if constexpr (d == Base::dim - 1)
            {
                return traverser_t<d>(m_set.get_traverser(index, d_ic), m_expansions[d]);
            }
            else
            {
                using Work     = MemoryPool<typename Set::template traverser_t<d>>;
                using WorkSize = typename Work::Size;

                Work& work = Work::getInstance();

                auto& offsetRange = std::get<d>(m_work_offsetRanges);

                const auto set_traversers_offsets = work.requestChunk(WorkSize(2 * (m_expansions[d + 1] + 1)));
                auto end_offset                   = set_traversers_offsets.first;

                xt::xtensor_fixed<value_t, xt::xshape<Base::dim - 1>> tmp_index(index);

                for (value_t width = 0; width != m_expansions[d + 1] + 1; ++width)
                {
                    tmp_index[d + 1] = index[d + 1] + width;
                    std::construct_at(work.getPtr(end_offset), m_set.get_traverser(tmp_index, d_ic));
                    if (work.at(end_offset).is_empty())
                    {
                        std::destroy_at(work.getPtr(end_offset));
                    }
                    else
                    {
                        ++end_offset;
                    }

                    tmp_index[d + 1] = index[d + 1] - width;
                    std::construct_at(work.getPtr(end_offset), m_set.get_traverser(tmp_index, d_ic));
                    if (work.at(end_offset).is_empty())
                    {
                        std::destroy_at(work.getPtr(end_offset));
                    }
                    else
                    {
                        ++end_offset;
                    }
                }

                offsetRange.push_back(set_traversers_offsets);

                return traverser_t<d>(set_traversers_offsets.first, end_offset, m_expansions[d]);
            }
        }

      private:

        Set m_set;
        expansion_t m_expansions;

        mutable OffsetRangeWork m_work_offsetRanges;
    };

    template <class Set>
    auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::contraction_t& expansions)
    {
        return Expansion(self(set), expansions);
    }

    template <class Set>
    auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::value_t& expansion)
    {
        return Expansion(self(set), expansion);
    }

    template <class Set>
    auto expand(const Set& set,
                const typename Contraction<std::decay_t<decltype(self(set))>>::value_t& expansion,
                const typename Contraction<std::decay_t<decltype(self(set))>>::do_expansion_t& do_expansion)
    {
        return Expansion(self(set), expansion, do_expansion);
    }

} // namespace samurai
