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

        hold(Mesh& mesh)
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

        inner_mesh_type(mesh_t& mesh)
            : p_mesh(&mesh)
        {
        }

        const mesh_t& mesh() const
        {
            return *p_mesh;
        }

        mesh_t& mesh()
        {
            return *p_mesh;
        }

        const mesh_t* mesh_ptr() const
        {
            return p_mesh;
        }

        mesh_t* mesh_ptr()
        {
            return p_mesh;
        }

      private:

        mesh_t* p_mesh = nullptr;
    };

    template <class Mesh>
    class inner_mesh_type<hold<Mesh>>
    {
      public:

        using mesh_t = Mesh;

        inner_mesh_type() = default;

        inner_mesh_type(hold<Mesh>& mesh)
            : m_mesh(mesh.get())
        {
        }

        const mesh_t& mesh() const
        {
            return m_mesh;
        }

        mesh_t& mesh()
        {
            return m_mesh;
        }

        const mesh_t* mesh_ptr() const
        {
            return &m_mesh;
        }

        mesh_t* mesh_ptr()
        {
            return &m_mesh;
        }

      private:

        mesh_t m_mesh;
    };

}
