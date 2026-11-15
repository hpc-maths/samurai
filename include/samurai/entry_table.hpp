// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <any>
#include <array>
#include <atomic>
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
//   (level, direction, layer...);
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
//     samurai::for_each_entry(mesh, my_key{level}, [&](const auto& run) { ... });
//
// for_each_entry runs the kernel on the entries of the table when the mesh
// keeps one, and on the entries produce() emits otherwise: the kernel is the
// same, only where the entries come from changes.
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

    // For entries with a variable number of cells: `count` storage indices from
    // `first` in the pool of the table. A producer fills it with
    //     PoolRange range{emit.pool().size(), 0};
    //     emit.pool().push_back(...); ++range.count;
    // and the kernel takes the pool as second argument:
    //     [&](const auto& entry, std::span<const std::size_t> pool) { ... }
    struct PoolRange
    {
        std::size_t first;
        std::size_t count;
    };

    // A key is compared and hashed through its bytes: it must not have padding.
    template <class Key>
    concept entry_table_key = requires { typename Key::entry_type; } && std::is_trivially_copyable_v<typename Key::entry_type>
                           && std::is_trivially_copyable_v<Key> && std::has_unique_object_representations_v<Key>;

    template <class Entry>
    struct EntryTable
    {
        std::vector<Entry> entries;
        std::vector<std::size_t> pool;
    };

    // Whether meshes keep tables. Set SAMURAI_ENTRY_TABLES=0 to always traverse
    // the set expressions.
    inline bool& entry_tables_enabled()
    {
        static bool enabled = []
        {
            const char* env = std::getenv("SAMURAI_ENTRY_TABLES");
            return env == nullptr || std::string_view(env) != "0";
        }();
        return enabled;
    }

    namespace detail
    {
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

        // What produce() emits to when the entries are kept.
        template <class Entry>
        class EntryRecorder
        {
          public:

            explicit EntryRecorder(EntryTable<Entry>& table)
                : m_table(table)
            {
            }

            void operator()(const Entry& entry)
            {
                m_table.entries.push_back(entry);
            }

            std::vector<std::size_t>& pool()
            {
                return m_table.pool;
            }

          private:

            EntryTable<Entry>& m_table;
        };

        // What produce() emits to when they are not: each entry goes straight to the kernel.
        template <class Entry, class Kernel>
        class EntryForwarder
        {
          public:

            explicit EntryForwarder(Kernel& kernel)
                : m_kernel(kernel)
            {
            }

            void operator()(const Entry& entry)
            {
                call_kernel(m_kernel, entry, std::span<const std::size_t>(m_pool));
                m_pool.clear();
            }

            std::vector<std::size_t>& pool()
            {
                return m_pool;
            }

          private:

            Kernel& m_kernel;
            std::vector<std::size_t> m_pool;
        };

        struct KeyBytesHash
        {
            template <class Key>
            std::size_t operator()(const Key& key) const noexcept
            {
                // FNV-1a over the words of the key, its last bytes zero padded
                std::array<std::size_t, (sizeof(Key) + sizeof(std::size_t) - 1) / sizeof(std::size_t)> words{};
                std::memcpy(words.data(), &key, sizeof(Key));
                std::size_t h = 14695981039346656037ULL;
                for (const auto w : words)
                {
                    h = (h ^ w) * 1099511628211ULL;
                    h ^= h >> 29;
                }
                return h;
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

        // A key can refuse to be kept on a mesh: bool keep(const Mesh&) const.
        template <class Key, class Mesh>
        bool keeps_table(const Key& key, const Mesh& mesh)
        {
            if constexpr (requires {
                              { key.keep(mesh) } -> std::convertible_to<bool>;
                          })
            {
                return key.keep(mesh);
            }
            else
            {
                return true;
            }
        }
    }

    // The entries of `key` on `mesh`, from its set expression, not kept.
    template <class Mesh, entry_table_key Key>
    auto record_entries_by_traversal(const Mesh& mesh, const Key& key)
    {
        EntryTable<typename Key::entry_type> table;
        detail::EntryRecorder<typename Key::entry_type> recorder(table);
        key.produce(mesh, recorder);
        return table;
    }

    // A key can also give its entries by a direct walk of the mesh, cheaper
    // than the set expression but allowed to give up:
    //     bool walk(const Mesh& mesh, Emit& emit) const; // false: use produce()
    // It must emit the entries of produce(), in the same order. It is only
    // used to record a table.
    template <class Key, class Mesh>
    concept walkable_key = requires(const Key& key, const Mesh& mesh, detail::EntryRecorder<typename Key::entry_type>& recorder) {
        { key.walk(mesh, recorder) } -> std::convertible_to<bool>;
    };

    // The entries of `key` on `mesh`, as a table records them: from the walk
    // of the key when it has one that succeeds, from its set expression otherwise.
    template <class Mesh, entry_table_key Key>
    auto record_entries(const Mesh& mesh, const Key& key)
    {
        if constexpr (walkable_key<Key, Mesh>)
        {
            EntryTable<typename Key::entry_type> table;
            detail::EntryRecorder<typename Key::entry_type> recorder(table);
            if (key.walk(mesh, recorder))
            {
                return table;
            }
        }
        return record_entries_by_traversal(mesh, key);
    }

    // The tables of one mesh, each recorded the first time it is asked for. A
    // mesh owns one and empties it whenever its cells change, so a table it
    // holds is never stale. It knows nothing of the keys.
    class EntryTableStore
    {
      public:

        EntryTableStore() = default;

        // The tables are not copied: the copy records its own on first use.
        EntryTableStore(const EntryTableStore&) noexcept
        {
        }

        EntryTableStore& operator=(const EntryTableStore& other) noexcept
        {
            if (this != &other)
            {
                clear();
            }
            return *this;
        }

        EntryTableStore(EntryTableStore&& other) noexcept
            : m_tables(std::move(other.m_tables))
            , m_size(std::exchange(other.m_size, 0))
        {
            other.m_tables.clear();
        }

        EntryTableStore& operator=(EntryTableStore&& other) noexcept
        {
            m_tables = std::move(other.m_tables);
            m_size   = std::exchange(other.m_size, 0);
            other.m_tables.clear();
            return *this;
        }

        ~EntryTableStore() = default;

        void swap(EntryTableStore& other) noexcept
        {
            m_tables.swap(other.m_tables);
            std::swap(m_size, other.m_size);
        }

        void clear() noexcept
        {
            const std::unique_lock lock(m_mutex);
            m_tables.clear();
            m_size = 0;
        }

        // Number of tables recorded.
        std::size_t size() const
        {
            const std::shared_lock lock(m_mutex);
            return m_size;
        }

        // The table of `key`, recorded on `mesh` the first time. Safe to call
        // from several threads: the reference stays valid until clear().
        template <entry_table_key Key, class Mesh>
        const EntryTable<typename Key::entry_type>& get(const Key& key, const Mesh& mesh) const
        {
            using tables_t = std::unordered_map<Key, EntryTable<typename Key::entry_type>, detail::KeyBytesHash, detail::KeyBytesEqual>;
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
            const auto [it, added] = std::any_cast<tables_t&>(slot).try_emplace(key, std::move(table));
            m_size += added ? 1 : 0;
            return it->second;
        }

      private:

        mutable std::shared_mutex m_mutex;

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

        // m_tables[type_id<Key>()] holds the tables of the keys of type Key.
        mutable std::vector<std::any> m_tables;
        mutable std::size_t m_size = 0;
    };

    template <class Mesh>
    concept has_entry_tables = requires(const Mesh& mesh) {
        { mesh.entry_tables() } -> std::same_as<const EntryTableStore&>;
    };

    // Whether for_each_entry reads the entries of `key` from a table of `mesh`.
    template <class Mesh, entry_table_key Key>
    bool has_entry_table(const Mesh& mesh, const Key& key)
    {
        if constexpr (has_entry_tables<Mesh>)
        {
            return entry_tables_enabled() && detail::keeps_table(key, mesh);
        }
        else
        {
            return false;
        }
    }

    // Calls kernel(entry), or kernel(entry, pool), for each entry of `key` on
    // `mesh`, in the order produce() emits them: from the table of the mesh when
    // it keeps one, by traversing the set expression otherwise.
    template <class Mesh, entry_table_key Key, class Kernel>
    void for_each_entry(const Mesh& mesh, const Key& key, Kernel&& kernel)
    {
        using entry_t = typename Key::entry_type;
        if constexpr (has_entry_tables<Mesh>)
        {
            if (has_entry_table(mesh, key))
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
        detail::EntryForwarder<entry_t, std::remove_reference_t<Kernel>> forwarder(kernel);
        key.produce(mesh, forwarder);
    }

    // The runs of `len` cells along x from each of the given cells:
    //     make_rows(mesh, i.size(), {{level, i.start, index}, {level + 1, 2 * i.start, 2 * index}})
    template <class Mesh, std::size_t N>
    auto
    make_rows(const Mesh& mesh, std::size_t len, const detail::CellCoord<Mesh::dim, typename Mesh::interval_t::value_t> (&first_cells)[N])
    {
        Rows<N> run{len, {}};
        for (std::size_t k = 0; k < N; ++k)
        {
            run.at[k] = memory_offset(mesh, first_cells[k]);
        }
        return run;
    }
}
