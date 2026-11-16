// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <any>
#include <array>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils.hpp"

// Entry tables: the traversal of a set expression kept as a flat list of
// entries, so that a kernel applied again and again on the same mesh does not
// traverse the set expression each time.
//
// A table is named by a key, a small struct that
// - gives the type of the entries: `using entry_type = ...;`, any trivially
//   copyable struct (Rows and Cells below cover the usual cases);
// - holds, as members, everything the traversal depends on besides the mesh
//   (level, direction, layer...). Two keys are the same when their bytes are, so
//   a key is trivially copyable and has no padding: members of one integer
//   type, std::size_t for instance, and direction_code() for a direction;
// - knows how to produce the entries, by traversing the set expression:
//       template <class Mesh, class Emit>
//       void produce(const Mesh& mesh, Emit& emit) const; // calls emit(entry)
//
//     struct my_key
//     {
//         using entry_type = samurai::Rows<2>;
//         std::size_t level;
//
//         void produce(const auto& mesh, auto& emit) const
//         {
//             auto set = ...; // depends on mesh and level only
//             set([&](const auto& i, const auto& index)
//                 {
//                     emit(samurai::make_rows(mesh, i.size(), {{level, i.start, index}, {level, i.start + 1, index}}));
//                 });
//         }
//     };
//
//     samurai::for_each_entry(mesh, my_key{level}, [&](const auto& rows) { ... });
//
// for_each_entry runs the kernel on the entries of the table when the mesh
// keeps one, and on the entries produce() emits otherwise: the kernel is the
// same, only where the entries come from changes.
//
// An entry with a variable number of cells holds a PoolRange:
//     auto cells = emit.new_range();
//     emit.add(cells, index); // as many times as needed
//     emit(my_entry{..., cells});
// and its kernel takes the pool as second argument:
//     [&](const auto& entry, std::span<const std::size_t> pool)
//     {
//         for (const auto index : samurai::cells_of(pool, entry.cells)) { ... }
//     }
//
// A key can also give its entries by a direct walk of the mesh, cheaper than
// the set expression but allowed to give up:
//     bool walk(const Mesh& mesh, Emit& emit) const; // false: use produce()
// It must emit the entries of produce(), in the same order. It is only used to
// record a table.
namespace samurai
{
    // N rows of `len` consecutive cells, each given by the storage index of its first cell.
    template <std::size_t N>
    struct Rows
    {
        std::size_t len;
        std::array<std::size_t, N> at;
    };

    // N cells, given by their storage index.
    template <std::size_t N>
    using Cells = std::array<std::size_t, N>;

    // `count` storage indices from `first` in the pool of the table.
    struct PoolRange
    {
        std::size_t first;
        std::size_t count;
    };

    // The storage indices of `range`.
    inline std::span<const std::size_t> cells_of(std::span<const std::size_t> pool, const PoolRange& range)
    {
        return pool.subspan(range.first, range.count);
    }

    // Whether meshes keep tables. Set the environment variable
    // SAMURAI_ENTRY_TABLES to 0 to always traverse the set expressions.
    inline bool& entry_tables_enabled()
    {
        static bool enabled = []
        {
            const char* value = std::getenv("SAMURAI_ENTRY_TABLES");
            return value == nullptr || std::string_view(value) != "0";
        }();
        return enabled;
    }

    namespace detail
    {
        template <class Key>
        constexpr void check_entry_table_key()
        {
            static_assert(requires { typename Key::entry_type; }, "an entry table key gives the type of its entries: using entry_type = ...;");
            static_assert(std::is_trivially_copyable_v<typename Key::entry_type>, "the entries of an entry table are trivially copyable");
            static_assert(
                std::is_trivially_copyable_v<Key> && std::has_unique_object_representations_v<Key>,
                "an entry table key is compared through its bytes: trivially copyable, without padding (members of one integer type)");
        }

        template <class Entry>
        struct EntryTable
        {
            std::vector<Entry> entries;
            std::vector<std::size_t> pool;
        };

        template <class Kernel, class Entry>
        void call_kernel(Kernel& kernel, const Entry& entry, std::span<const std::size_t> pool)
        {
            if constexpr (std::invocable<Kernel&, const Entry&, std::span<const std::size_t>>)
            {
                kernel(entry, pool);
            }
            else
            {
                kernel(entry);
            }
        }

        // The ranges a producer opens and fills in a pool.
        class PoolWriter
        {
          public:

            explicit PoolWriter(std::vector<std::size_t>& pool)
                : m_pool(pool)
            {
            }

            PoolRange new_range() const
            {
                return {m_pool.size(), 0};
            }

            // Appends to the range opened last.
            void add(PoolRange& range, std::size_t index)
            {
                m_pool.push_back(index);
                ++range.count;
            }

          private:

            std::vector<std::size_t>& m_pool;
        };

        // What produce() emits to when the entries are kept.
        template <class Entry>
        class EntryRecorder : public PoolWriter
        {
          public:

            explicit EntryRecorder(EntryTable<Entry>& table)
                : PoolWriter(table.pool)
                , m_table(table)
            {
            }

            void operator()(const Entry& entry)
            {
                m_table.entries.push_back(entry);
            }

          private:

            EntryTable<Entry>& m_table;
        };

        // What produce() emits to when they are not: each entry goes straight
        // to the kernel, with the pool of that entry alone.
        template <class Entry, class Kernel>
        class EntryForwarder : public PoolWriter
        {
          public:

            EntryForwarder(Kernel& kernel, std::vector<std::size_t>& pool)
                : PoolWriter(pool)
                , m_kernel(kernel)
                , m_pool(pool)
            {
            }

            void operator()(const Entry& entry)
            {
                call_kernel(m_kernel, entry, std::span<const std::size_t>(m_pool));
                m_pool.clear();
            }

          private:

            Kernel& m_kernel;
            std::vector<std::size_t>& m_pool;
        };

        struct KeyBytesHash
        {
            template <class Key>
            std::size_t operator()(const Key& key) const noexcept
            {
                // A key has no padding (see check_entry_table_key), so its bytes
                // are its value: hashed as a string of bytes by the standard library.
                const auto bytes = std::bit_cast<std::array<char, sizeof(Key)>>(key);
                return std::hash<std::string_view>{}(std::string_view(bytes.data(), bytes.size()));
            }
        };

        struct KeyBytesEqual
        {
            template <class Key>
            bool operator()(const Key& a, const Key& b) const noexcept
            {
                return std::memcmp(&a, &b, sizeof(Key)) == 0;
            }
        };

        template <class Key, class Mesh>
        concept walkable_key = requires(const Key& key, const Mesh& mesh, EntryRecorder<typename Key::entry_type>& recorder) {
            { key.walk(mesh, recorder) } -> std::convertible_to<bool>;
        };

        // The entries of `key` on `mesh`: from the walk of the key when it has
        // one, `by_walk` allows it and it succeeds, from its set expression otherwise.
        template <class Mesh, class Key>
        auto record_entries(const Mesh& mesh, const Key& key, bool by_walk = true)
        {
            EntryTable<typename Key::entry_type> table;
            EntryRecorder<typename Key::entry_type> recorder(table);
            if constexpr (walkable_key<Key, Mesh>)
            {
                if (by_walk && key.walk(mesh, recorder))
                {
                    return table;
                }
                table = {};
            }
            key.produce(mesh, recorder);
            return table;
        }

        // The tables of one mesh, each recorded the first time it is asked for.
        // It knows nothing of the keys. A mesh owns one and empties it each
        // time it rebuilds its cells or the meshes of its neighbours (see
        // Mesh_base): a table is valid as long as the mesh is not modified
        // behind its back, through its non-const accessors.
        //
        // Recording behind a const mesh makes it safe to read the tables of one
        // mesh from several threads, as the prediction caches are.
        class EntryTableStore
        {
          public:

            EntryTableStore() = default;

            // A copy starts empty and records its tables again on first use: they
            // are a cache, and std::shared_mutex is not copyable, so the copy of a
            // mesh would be deleted without this constructor.
            EntryTableStore(const EntryTableStore&) noexcept
            {
            }

            // Same for an assignment: the store assigned to is emptied, not
            // filled with the tables of `other`.
            EntryTableStore& operator=(const EntryTableStore& other) noexcept
            {
                if (this != &other)
                {
                    clear();
                }
                return *this;
            }

            EntryTableStore(EntryTableStore&& other) noexcept
            {
                swap(other);
            }

            EntryTableStore& operator=(EntryTableStore&& other) noexcept
            {
                if (this != &other)
                {
                    clear();
                    swap(other);
                }
                return *this;
            }

            ~EntryTableStore() = default;

            void swap(EntryTableStore& other) noexcept
            {
                if (this != &other)
                {
                    const std::scoped_lock lock(m_mutex, other.m_mutex);
                    m_tables.swap(other.m_tables);
                }
            }

            void clear() noexcept
            {
                const std::unique_lock lock(m_mutex);
                m_tables.clear();
            }

            // The table of `key`, recorded on `mesh` the first time. The
            // reference stays valid until clear().
            template <class Key, class Mesh>
            const EntryTable<typename Key::entry_type>& get(const Key& key, const Mesh& mesh) const
            {
                using tables_t         = std::unordered_map<Key, EntryTable<typename Key::entry_type>, KeyBytesHash, KeyBytesEqual>;
                const std::size_t type = type_id<Key>();
                {
                    const std::shared_lock lock(m_mutex);
                    if (type < m_tables.size() && m_tables[type].has_value())
                    {
                        const auto& tables = std::any_cast<const tables_t&>(m_tables[type]);
                        if (const auto found = tables.find(key); found != tables.end())
                        {
                            return found->second;
                        }
                    }
                }
                // Recorded outside the lock: a producer may read other tables. When
                // two threads record the same table, the first one kept wins.
                auto table = record_entries(mesh, key);
                const std::unique_lock lock(m_mutex);
                if (type >= m_tables.size())
                {
                    m_tables.resize(type + 1);
                }
                auto& slot = m_tables[type];
                if (!slot.has_value())
                {
                    slot.emplace<tables_t>();
                }
                return std::any_cast<tables_t&>(slot).try_emplace(key, std::move(table)).first->second;
            }

          private:

            // One number per key type, in the order the types are first asked for.
            template <class Key>
            static std::size_t type_id()
            {
                static const std::size_t id = next_type_id()++;
                return id;
            }

            static std::atomic<std::size_t>& next_type_id()
            {
                static std::atomic<std::size_t> next{0};
                return next;
            }

            mutable std::shared_mutex m_mutex;

            // m_tables[type_id<Key>()] holds the tables of the keys of type Key.
            mutable std::vector<std::any> m_tables;
        };

        template <class Mesh>
        concept has_entry_tables = requires(const Mesh& mesh) {
            { mesh.entry_tables() } -> std::same_as<const EntryTableStore&>;
        };
    }

    // Calls kernel(entry), or kernel(entry, pool), for each entry of `key` on
    // `mesh`, in the order produce() emits them: from the table of the mesh when
    // it keeps tables, by traversing the set expression otherwise.
    template <class Mesh, class Key, class Kernel>
    void for_each_entry(const Mesh& mesh, const Key& key, Kernel&& kernel)
    {
        detail::check_entry_table_key<Key>();
        if constexpr (detail::has_entry_tables<Mesh>)
        {
            if (entry_tables_enabled())
            {
                const auto& table = mesh.entry_tables().get(key, mesh);
                const std::span<const std::size_t> pool(table.pool);
                for (const auto& entry : table.entries)
                {
                    detail::call_kernel(kernel, entry, pool);
                }
                return;
            }
        }
        std::vector<std::size_t> pool;
        detail::EntryForwarder<typename Key::entry_type, std::remove_reference_t<Kernel>> forwarder(kernel, pool);
        key.produce(mesh, forwarder);
    }

    // The rows of `len` cells along x from each of the given cells:
    //     make_rows(mesh, i.size(), {{level, i.start, index}, {level + 1, 2 * i.start, 2 * index}})
    template <class Mesh, std::size_t N>
    auto
    make_rows(const Mesh& mesh, std::size_t len, const detail::CellCoord<Mesh::dim, typename Mesh::interval_t::value_t> (&first_cells)[N])
    {
        Rows<N> rows{len, {}};
        for (std::size_t k = 0; k < N; ++k)
        {
            rows.at[k] = memory_offset(mesh, first_cells[k]);
        }
        return rows;
    }
}
