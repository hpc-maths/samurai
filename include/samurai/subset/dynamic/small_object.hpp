// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

// Small-buffer optimization for a value-semantic, type-erased object.
//
// This is the whole (and only) place where placement-new lives. It lets a
// polymorphic object be stored *inside* its holder when it is small enough,
// and on the heap otherwise, while the holder keeps clean value semantics
// (deep copy / move). No raw owning `new`/`delete`: the inline case uses
// placement-new (which allocates nothing) plus a manual destructor call, and
// the heap fallback is a plain `std::unique_ptr`.
namespace samurai::sbo
{
    /**
     * Result of placing a copy/move of a polymorphic object: `ptr` always
     * points at the new object; `owner` holds it iff it went on the heap, and
     * is null when it was placement-constructed into the caller's buffer.
     */
    template <class Interface>
    struct PlacedObject
    {
        Interface* ptr = nullptr;
        std::unique_ptr<Interface> owner;
    };

    /**
     * CRTP mixin that gives a concrete `Derived : Interface` the `copy_to` /
     * `move_to` operations `SmallObject` needs. A concrete type placed in a
     * `SmallObject` should derive from `Cloneable<Interface, Derived>` instead
     * of directly from `Interface`.
     */
    template <class Interface, class Derived>
    class Cloneable : public Interface
    {
      public:

        PlacedObject<Interface> copy_to(void* buffer, std::size_t capacity) const final
        {
            if (fits(capacity))
            {
                Interface* placed = std::construct_at(static_cast<Derived*>(buffer), as_derived());
                return {placed, nullptr};
            }
            std::unique_ptr<Interface> owner = std::make_unique<Derived>(as_derived());
            Interface* placed                = owner.get();
            return {placed, std::move(owner)};
        }

        PlacedObject<Interface> move_to(void* buffer, std::size_t capacity) final
        {
            if (fits(capacity))
            {
                Interface* placed = std::construct_at(static_cast<Derived*>(buffer), std::move(as_derived()));
                return {placed, nullptr};
            }
            std::unique_ptr<Interface> owner = std::make_unique<Derived>(std::move(as_derived()));
            Interface* placed                = owner.get();
            return {placed, std::move(owner)};
        }

      private:

        static constexpr bool fits(std::size_t capacity)
        {
            return sizeof(Derived) <= capacity && alignof(Derived) <= alignof(std::max_align_t);
        }

        const Derived& as_derived() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& as_derived()
        {
            return static_cast<Derived&>(*this);
        }
    };

    /**
     * A value handle over a polymorphic `Interface` object, stored inline in a
     * `BufferSize`-byte buffer when it fits and on the heap otherwise. Copy and
     * move are deep. `Interface` must expose `copy_to` / `move_to`; derive the
     * concrete types from `Cloneable` to get them for free.
     */
    template <class Interface, std::size_t BufferSize>
    class SmallObject
    {
      public:

        template <class Concrete>
            requires std::derived_from<std::remove_cvref_t<Concrete>, Interface>
        explicit SmallObject(Concrete&& object)
        {
            using T = std::remove_cvref_t<Concrete>;
            if constexpr (sizeof(T) <= BufferSize && alignof(T) <= alignof(std::max_align_t))
            {
                m_ptr = std::construct_at(static_cast<T*>(static_cast<void*>(m_storage)), std::forward<Concrete>(object));
            }
            else
            {
                m_heap = std::make_unique<T>(std::forward<Concrete>(object));
                m_ptr  = m_heap.get();
            }
        }

        SmallObject(const SmallObject& other)
        {
            adopt(other.m_ptr->copy_to(m_storage, BufferSize));
        }

        SmallObject(SmallObject&& other) noexcept
        {
            steal_from(other);
        }

        SmallObject& operator=(const SmallObject& other)
        {
            if (this != &other)
            {
                reset();
                adopt(other.m_ptr->copy_to(m_storage, BufferSize));
            }
            return *this;
        }

        SmallObject& operator=(SmallObject&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                steal_from(other);
            }
            return *this;
        }

        ~SmallObject()
        {
            reset();
        }

        Interface* get()
        {
            return m_ptr;
        }

        const Interface* get() const
        {
            return m_ptr;
        }

      private:

        // Whatever `m_ptr` points at that is not owned by `m_heap` lives in the
        // inline buffer and must be destroyed by hand.
        bool is_inline() const
        {
            return m_ptr != nullptr && m_heap == nullptr;
        }

        void reset()
        {
            if (is_inline())
            {
                std::destroy_at(m_ptr);
            }
            m_heap.reset();
            m_ptr = nullptr;
        }

        void adopt(PlacedObject<Interface> placed)
        {
            m_ptr  = placed.ptr;
            m_heap = std::move(placed.owner);
        }

        void steal_from(SmallObject& other)
        {
            if (other.m_heap != nullptr)
            {
                // Heap object: transfer ownership. Nothing left to destroy in `other`.
                m_heap      = std::move(other.m_heap);
                m_ptr       = m_heap.get();
                other.m_ptr = nullptr;
            }
            else if (other.m_ptr != nullptr)
            {
                // Inline object: move-construct into our buffer, then destroy the source.
                adopt(other.m_ptr->move_to(m_storage, BufferSize));
                other.reset();
            }
        }

        alignas(std::max_align_t) std::byte m_storage[BufferSize];
        std::unique_ptr<Interface> m_heap;
        Interface* m_ptr = nullptr;
    };

} // namespace samurai::sbo
