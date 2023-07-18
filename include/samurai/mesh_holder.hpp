// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

namespace samurai
{

    template <class Mesh>
    class hold
    {
      public:

        static constexpr std::size_t dim = Mesh::dim;
        using interval_t                 = typename Mesh::interval_t;

        explicit hold(Mesh& mesh)
            : m_mesh(mesh)
        {
        }

        Mesh& get()
        {
            return m_mesh;
        }

      private:

        Mesh& m_mesh; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    };

    template <class Mesh>
    auto holder(Mesh& mesh)
    {
        return hold<Mesh>(mesh);
    }

    template <class Mesh>
    class inner_mesh_type
    {
      public:

        using mesh_t = Mesh;

        inner_mesh_type() = default;

        explicit inner_mesh_type(const mesh_t& mesh)
            : p_mesh(&(const_cast<mesh_t&>(mesh)))
        {
        }

        inner_mesh_type& operator=(const mesh_t& mesh)
        {
            p_mesh = &(const_cast<mesh_t&>(mesh));
            return *this;
        }

        const mesh_t& mesh() const
        {
            return *p_mesh;
        }

        mesh_t& mesh()
        {
            return *p_mesh;
        }

        void change_mesh_ptr(mesh_t& mesh)
        {
            p_mesh = &mesh;
        }

      public:

        mesh_t* p_mesh = nullptr;
    };

    template <class Mesh>
    class inner_mesh_type<hold<Mesh>>
    {
      public:

        using mesh_t = Mesh;

        inner_mesh_type() = default;

        explicit inner_mesh_type(hold<Mesh>& mesh)
            : m_mesh(mesh.get())
        {
        }

        explicit inner_mesh_type(const Mesh& mesh)
            : m_mesh(mesh)
        {
        }

        inner_mesh_type& operator=(const Mesh& mesh)
        {
            m_mesh = mesh;
            return *this;
        }

        const mesh_t& mesh() const
        {
            return m_mesh;
        }

        mesh_t& mesh()
        {
            return m_mesh;
        }

      private:

        mesh_t m_mesh;
    };

}
