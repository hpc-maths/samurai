// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_range_base.hpp"

namespace samurai
{

    template <class SetTraverserRange>
    class TranslationTraverserRange;

    template <class SetTraverserRange>
    struct SetTraverserRangeTraits<TranslationTraverserRange<SetTraverserRange>>
    {
        static_assert(IsSetTraverserRange<SetTraverserRange>::value);

        using ChildIterator       = typename SetTraverserRange::Iterator;
        using const_ChildIterator = typename SetTraverserRange::const_Iterator;

        using Translation = typename ChildIterator::value_type::value_type;

        template <class Inner_Iterator>
        class _Iterator
        {
          public:

            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = TranslationTraverser<typename Inner_Iterator::value_type>;
            using reference         = value_type;

            _Iterator(const Inner_Iterator inner_Iterator, const Translation& translation)
                : m_innerIterator(inner_Iterator)
                , m_translation(translation)
            {
            }

            reference operator*() const
            {
                return reference(*m_innerIterator, m_translation);
            }

            _Iterator& operator++()
            {
                ++m_innerIterator;
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
                return a.m_innerIterator == b.m_innerIterator;
            };

            friend bool operator!=(const _Iterator& a, const _Iterator& b)
            {
                return a.m_innerIterator != b.m_innerIterator;
            };

          private:

            Inner_Iterator m_innerIterator;
            Translation m_translation;
        };

        using Iterator       = _Iterator<ChildIterator>;
        using const_Iterator = _Iterator<const_ChildIterator>;
    };

    template <class SetTraverserRange>
    class TranslationTraverserRange : public SetTraverserRangeBase<TranslationTraverserRange<SetTraverserRange>>
    {
        using Self = TranslationTraverserRange<SetTraverserRange>;

      public:

        SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS

        using Translation = typename SetTraverserRangeTraits<Self>::Translation;

        TranslationTraverserRange(const SetTraverserRange& set_traverser_range, const Translation& translation)
            : m_set_traverser_range(set_traverser_range)
            , m_translation(translation)
        {
        }

        Iterator begin_impl()
        {
            return Iterator(m_set_traverser_range.begin(), m_translation);
        }

        Iterator end_impl()
        {
            return Iterator(m_set_traverser_range.end(), m_translation);
        }

        const_Iterator begin_impl() const
        {
            return const_Iterator(m_set_traverser_range.cbegin(), m_translation);
        }

        const_Iterator end_impl() const
        {
            return const_Iterator(m_set_traverser_range.cend(), m_translation);
        }

      private:

        SetTraverserRange m_set_traverser_range;
        Translation m_translation;
    };

} // namespace samurai
