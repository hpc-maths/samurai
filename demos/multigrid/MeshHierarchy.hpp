#include <samurai/amr/mesh.hpp>
#include <list>

template<class Mesh>
class MeshHierarchy
{
private:
    Mesh& _initial_mesh;
    std::vector<Mesh> _created_meshes;

    // front() --> fine mesh
    // back()  --> coarse mesh
    std::list<Mesh*> _hierarchy;
public:
    MeshHierarchy(Mesh& mesh) :
        _initial_mesh(mesh)
    {
        _hierarchy.push_front(&mesh);
    }

    Mesh& add_coarser()
    {
        auto mesh = _hierarchy.back();
        _created_meshes.emplace_back(coarsen(mesh));
        Mesh& coarser = _created_meshes.back();
        _hierarchy.push_back(&coarser);
        return coarser;
    }

    Mesh& add_finer()
    {
        auto mesh = _hierarchy.front();
        _created_meshes.emplace_back(refine(mesh));
        Mesh& finer = _created_meshes.back();
        _hierarchy.push_front(&finer);
        return finer;
    }

    void build_from_fine(int times)
    {
        for (int i=0; i<times; ++i)
            add_coarser();
    }

    void build_from_coarse(int times)
    {
        for (int i=0; i<times; ++i)
            add_finer();
    }

    Mesh* get_coarser(const Mesh& m)
    {
        for (auto it = begin(); it != end(); ++it)
        {
            if (*it == &m)
            {
                ++it;
                if (it != end())
                    return *it;
                else
                    return nullptr;
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

public:
    static Mesh coarsen(const Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        typename Mesh::cl_type coarse_cell_list;
        if (Mesh::dim == 1)
        {
            samurai::for_each_interval(mesh[mesh_id_t::cells], [&](size_t level, const auto& i, const auto& index)
            {
                coarse_cell_list[level-1][{}].add_interval(i >> 1);
            });
        }
        else if (Mesh::dim == 2)
        {
            samurai::for_each_interval(mesh[mesh_id_t::cells], [&](size_t level, const auto& i, const auto& index)
            {
                auto j = index[0];
                if (j % 2 == 0)
                    coarse_cell_list[level-1][{j/2}].add_interval(i/2);
            });
        }
        return Mesh(coarse_cell_list, mesh.min_level()-1, mesh.max_level()-1);
    }

    /*static void coarsen(const Mesh& mesh, Mesh& coarse_mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        typename Mesh::cl_type coarse_cell_list;
        samurai::for_each_interval(mesh[mesh_id_t::cells], [&](size_t l, const auto& interval, const auto& index)
        {
            coarse_cell_list[l-1][{}].add_interval(interval >> 1);
        });
        coarse_mesh = Mesh(coarse_cell_list, mesh.min_level()-1, mesh.max_level()-1);
    }*/

    static Mesh refine(const Mesh& mesh)
    {
        assert(false && "not implemented");
    }
};