// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{

    template <class... SetTraverserRanges>
    class UnionTraverserRange;

    template <class... SetTraverserRanges>
    struct SetTraverserRangeTraits<UnionTraverserRange<SetTraverserRanges...>>
    {
        static_assert((IsSetTraverserRange<SetTraverserRanges>::value and ...));

        template <class... Inner_Iterators>
        class _Iterator
        {
          public:

            using Childrens = std::tuple<Inner_Iterators...>;

            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = UnionTraverser<typename Inner_Iterators::value_type...>;
            using reference         = value_type;

            static constexpr std::size_t nIntervals = std::tuple_size<Childrens>::value;

            _Iterator(const Inner_Iterators... inner_Iterators)
                : m_inner_Iterator(inner_Iterators...)
            {
            }

            reference operator*() const
            {
                return std::apply(
                    [](const auto&... innerIterators) -> void
                    {
                        return reference((*innerIterators)...);
                    },
                    m_innerIterators);
            }

            _Iterator& operator++()
            {
                static_for<0, nIntervals>::apply(
                    [this](const auto i) -> void
                    {
                        ++std::get<i>(m_innerIterators);
                    });
                return *this;
            }

            _Iterator operator++(int)
            {
                _Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const _Iterator& a, const _Iterator& b)
            {
                return a.m_inner_Iterator == b.m_inner_Iterator;
            };

            friend bool operator!=(const _Iterator& a, const _Iterator& b)
            {
                return a.m_inner_Iterator != b.m_inner_Iterator;
            };

          private:

            Childrens m_innerIterators;
        };

        using Iterator       = _Iterator<typename SetTraverserRanges::Iterator...>;
        using const_Iterator = _Iterator<typename SetTraverserRanges::const_Iterator...>;
    };

    template <class... SetTraverserRanges>
    class UnionTraverserRange : public SetTraverserRangeBase<UnionTraverserRange<SetTraverserRanges...>>
    {
        using Self = TranslationTraverserRange<SetTraverserRange>;

      public:

        SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS

        using Childrens = std::tuple<SetTraverserRanges...>;

        TranslationTraverserRange(const SetTraverserRanges&... set_traverser_ranges)
            : m_set_traverser_ranges(set_traverser_ranges...)
        {
        }

        Iterator begin_impl()
        {
            return std::apply(
                [](auto&... innerIterators) -> Iterator
                {
                    return Iterator(innerIterators.begin()...);
                },
                m_set_traverser_ranges);
        }

        Iterator end_impl()
        {
            return std::apply(
                [](auto&... innerIterators) -> Iterator
                {
                    return Iterator(innerIterators.end()...);
                },
                m_set_traverser_ranges);
        }

        const_Iterator begin_impl() const
        {
            return std::apply(
                [](const auto&... innerIterators) -> const_Iterator
                {
                    return const_Iterator(innerIterators.cbegin()...);
                },
                m_set_traverser_ranges);
        }

        const_Iterator end_impl() const
        {
            return std::apply(
                [](const auto&... innerIterators) -> const_Iterator
                {
                    return const_Iterator(innerIterators.cend()...);
                },
                m_set_traverser_ranges);
        }

      private:

        Childrens m_set_traverser_ranges;
    };

} // namespace samurai
