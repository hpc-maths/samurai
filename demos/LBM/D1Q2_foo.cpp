#include <array>
#include <math.h>
#include <vector>

#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mr/coarsening.hpp>
#include <mure/mr/refinement.hpp>
#include <mure/mr/mesh.hpp>
#include <mure/mr/mr_config.hpp>
#include <mure/mr/pred_and_proj.hpp>

template<class TInterval>
class projection_lbm_op_ : public mure::field_operator_base<TInterval> {
  public:
    INIT_OPERATOR(projection_lbm_op_)

    template<class T>
    void operator()(mure::Dim<1>, T &field) const
    {
        field(level, i) =
            .5 * (field(level + 1, 2 * i) + field(level + 1, 2 * i + 1));
    }
};

template<class T>
auto projection_lbm(T &&field)
{
    return mure::make_field_operator_function<projection_lbm_op_>(
        std::forward<T>(field));
}

template<class TInterval>
class prediction_lbm_op_ : public mure::field_operator_base<TInterval> {
  public:
    INIT_OPERATOR(prediction_lbm_op_)

    template<class T>
    void operator()(mure::Dim<1>, T &field) const
    {
        field(level, i) = field(level - 1, i>>1);
    }
};

template<class T>
auto prediction_lbm(T &&field)
{
    return mure::make_field_operator_function<prediction_lbm_op_>(
        std::forward<T>(field));
}

template<class Config>
void init_u(std::vector<mure::Field<Config>> &u, mure::Mesh<Config> &mesh)
{
    for (auto &uu : u)
        uu.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        u[0][cell] = exp(-200.0 * x * x);
        u[1][cell] = exp(-200.0 * x * x);
    });
}

template<class Config>
void one_time_step(std::vector<mure::Field<Config>> &u)
{
    double lambda = 1.;
    auto mesh = u[0].mesh();
    std::size_t nvel = u.size();
    auto max_level = mesh[mure::MeshType::cells].max_level();
    auto min_level = mesh[mure::MeshType::cells].min_level();
    auto min_union = mesh[mure::MeshType::union_cells].min_level();

    for (std::size_t level = min_level+1; level <= max_level; ++level)
    {
        auto subset_right =
                mure::intersection(
                    mure::difference(
                        mure::translate_in_x<1>(mesh[mure::MeshType::cells][level]),
                        mesh[mure::MeshType::cells][level]),
                    mesh[mure::MeshType::cells][level-1]).on(level);

        for (std::size_t i = 0; i < nvel; ++i)
        {
            subset_right.apply_op(level, prediction_lbm(u[i]));
        }

        auto subset_left =
                mure::intersection(
                    mure::difference(
                        mure::translate_in_x<-1>(mesh[mure::MeshType::cells][level]),
                        mesh[mure::MeshType::cells][level]),
                    mesh[mure::MeshType::cells][level-1]).on(level);

        for (std::size_t i = 0; i < nvel; ++i)
        {
            subset_left.apply_op(level, prediction_lbm(u[i]));
        }

        // auto exp =
        //     mure::intersection(mesh[mure::MeshType::cells_and_ghosts][level],
        //                        mesh[mure::MeshType::cells][level - 1])
        //         .on(level);
        // for (std::size_t i = 0; i < nvel; ++i)
        // {
        //     exp.apply_op(level, prediction_lbm(u[i]));
        // }
    }

    for (std::size_t level = 0; level < max_level; ++level)
    {
        auto subset_right =
                mure::intersection(
                    mure::difference(
                        mure::translate_in_x<1>(mesh[mure::MeshType::cells][level]),
                        mesh[mure::MeshType::cells][level]),
                    mesh[mure::MeshType::cells][level+1]).on(level);

        for (std::size_t i = 0; i < nvel; ++i)
        {
            subset_right.apply_op(level, projection_lbm(u[i]));
        }

        auto subset_left =
                mure::intersection(
                    mure::difference(
                        mure::translate_in_x<-1>(mesh[mure::MeshType::cells][level]),
                        mesh[mure::MeshType::cells][level]),
                    mesh[mure::MeshType::cells][level+1]).on(level);

        for (std::size_t i = 0; i < nvel; ++i)
        {
            subset_left.apply_op(level, projection_lbm(u[i]));
        }

        // auto exp =
        //     mure::intersection(mesh[mure::MeshType::cells_and_ghosts][level],
        //                        mesh[mure::MeshType::cells][level + 1])
        //         .on(level);
        // for (std::size_t i = 0; i < nvel; ++i)
        // {
        //     exp.apply_op(level, projection_lbm(u[i]));
        // }
    }

    std::vector<mure::Field<Config>> new_u;
    for (std::size_t i = 0; i < nvel; ++i)
    {
        std::stringstream str;
        str << "newu_" << i;
        new_u.push_back({str.str().data(), mesh});
        new_u[i].array().fill(0);
    }
    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);
        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];
            std::size_t j = max_level - level;
            double coeff = 1. / (1 << j);
            // 0.5 just to see why it doesnt work.
            new_u[0](level, i) = 1.0*(
                (1 - coeff) * u[0](level, i) +
                .5 * coeff *
                    (u[0](level, i - 1) + u[0](level, i + 1) +
                     1 / lambda * (u[1](level, i - 1) - u[1](level, i + 1))));
            new_u[1](level, i) = (
                (1 - coeff) * u[1](level, i) +
                .5 * coeff *
                    (lambda * (u[0](level, i - 1) - u[0](level, i + 1)) +
                     u[1](level, i - 1) + u[1](level, i + 1)));
        });
    }
    // It seems not working.
    for (std::size_t i = 0; i < nvel; ++i)
    {
        u[i].array() = new_u[i].array();
    }
}
// The input arguments are supposed to be the following:
// test case | maximum level | eps
int main(int argc, char *argv[])
{

    if (argc <= 2)
    {
        std::cerr << "Provide the number of the maximum level "
                     "and the epsilon, please!"
                  << std::endl;
        return -1;
    }

    constexpr size_t dim = 1;
    using Config = mure::MRConfig<dim>;

    mure::Box<double, dim> box({-2}, {2});
    mure::Mesh<Config> mesh{box, atoi(argv[1])};

    std::size_t nvel = 2;

    std::vector<mure::Field<Config>> u;
    for (std::size_t i = 0; i < nvel; ++i)
    {
        std::stringstream str;
        str << "u_" << i;
        u.push_back({str.str().data(), mesh});
    }

    // Initialization
    init_u(u, mesh);

    // double eps = 1e-2;
    double eps = atof(argv[2]);

    for (std::size_t ite = 0; ite < 20; ++ite)
    {
        std::vector<mure::Field<Config>> detail;
        for (std::size_t i = 0; i < nvel; ++i)
        {
            std::stringstream str;
            str << "detail_" << i;
            detail.push_back({str.str().data(), mesh});
            detail[i].array().fill(0);
            mure::mr_projection(u[i]);
        }
        mure::coarsening(detail, u, eps, ite);
    }

    {
    std::stringstream str;
    str << "LBM_D1Q2_init_lmax-" << argv[1] << "_eps-" << argv[2];
    auto h5file = mure::Hdf5(str.str().data());
    auto mesh_field = u[0].mesh();
    h5file.add_mesh(mesh_field);
    mure::Field<Config> level_{"level", mesh_field};
    mesh_field.for_each_cell(
        [&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(u[0]);
    h5file.add_field(u[1]);
    h5file.add_field(level_);
    }
    for (std::size_t nb_ite = 0; nb_ite < 100; ++nb_ite)
    {
        // for (std::size_t i = 0; i < nvel; ++i)
        // {
        //     mure::mr_projection(u[i]);
        //     mure::mr_prediction(u[i]);
        // }
        one_time_step(u);
        std::cout<<"Iteration = "<<nb_ite<<std::endl;
        // coarsening
        for (std::size_t ite = 0; ite < 10; ++ite)
        {
            std::vector<mure::Field<Config>> detail;
            for (std::size_t i = 0; i < nvel; ++i)
            {
                std::stringstream str;
                str << "detail_" << i;
                detail.push_back({str.str().data(), mesh});
                detail[i].array().fill(0);
                mure::mr_projection(u[i]);
            }
            mure::coarsening(detail, u, eps, ite);
        }

        // refinement
        /*
        for (std::size_t ite = 0; ite < 10; ++ite)
        {
            std::vector<mure::Field<Config>> detail;
            for (std::size_t i = 0; i < nvel; ++i)
            {
                std::stringstream str;
                str << "detail_" << i;
                detail.push_back({str.str().data(), mesh});
                detail[i].array().fill(0);
                mure::mr_projection(u[i]);
                mure::mr_prediction(u[i]);
            }
            mure::refinement(detail, u, eps);
        } */
    }

    std::stringstream str;
    str << "LBM_D1Q2_lmax-" << argv[1] << "_eps-" << argv[2];
    auto h5file = mure::Hdf5(str.str().data());
    auto mesh_field = u[0].mesh();
    h5file.add_mesh(mesh_field);
    mure::Field<Config> level_{"level", mesh_field};
    mesh_field.for_each_cell(
        [&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(u[0]);
    h5file.add_field(u[1]);
    h5file.add_field(level_);

    return 0;
}
