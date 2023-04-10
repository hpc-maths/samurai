#pragma once
#include "coarsening.hpp"
#include <list>

template <class Mesh>
class MeshHierarchy
{
  private:

    Mesh& _initial_mesh;
    std::vector<Mesh> _created_meshes;

    // front() --> fine mesh
    // back()  --> coarse mesh
    std::list<Mesh*> _hierarchy;

  public:

    MeshHierarchy(Mesh& mesh)
        : _initial_mesh(mesh)
    {
        _hierarchy.push_front(&mesh);
    }

    Mesh& add_coarser()
    {
        auto mesh = _hierarchy.back();
        _created_meshes.emplace_back(samurai_new::coarsen(mesh));
        Mesh& coarser = _created_meshes.back();
        _hierarchy.push_back(&coarser);
        return coarser;
    }

    Mesh& add_finer()
    {
        auto mesh = _hierarchy.front();
        _created_meshes.emplace_back(samurai_new::refine(mesh));
        Mesh& finer = _created_meshes.back();
        _hierarchy.push_front(&finer);
        return finer;
    }

    void build_from_fine(int times)
    {
        for (int i = 0; i < times; ++i)
        {
            add_coarser();
        }
    }

    void build_from_coarse(int times)
    {
        for (int i = 0; i < times; ++i)
        {
            add_finer();
        }
    }

    Mesh* get_coarser(const Mesh& m)
    {
        for (auto it = begin(); it != end(); ++it)
        {
            if (*it == &m)
            {
                ++it;
                if (it != end())
                {
                    return *it;
                }
                else
                {
                    return nullptr;
                }
            }
        }
        assert(false && "the mesh is not in the hierarchy");
        return nullptr;
    }

    // begin from the fine mesh
    auto begin()
    {
        return _hierarchy.begin();
    }

    auto end()
    {
        return _hierarchy.end();
    }

    Mesh& fine()
    {
        return _hierarchy.front();
    }

    Mesh& coarse()
    {
        return _hierarchy.back();
    }
};
