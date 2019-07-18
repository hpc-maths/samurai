#pragma once

#include <type_traits>

#include <xtensor/xexpression.hpp>
#include <xtl/xtype_traits.hpp>

#include "../level_cell_array.hpp"

namespace mure
{
    /**********************
     * node_op definition *
     **********************/

    template<class D>
    class node_op {
      public:
        using derived_type = D;

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;

        auto index(int i) const noexcept;
        auto size(std::size_t dim) const noexcept;
        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;
        auto offset(std::size_t dim, std::size_t index) const noexcept;
        auto interval(std::size_t dim, std::size_t index) const noexcept;
        auto offsets_size(std::size_t dim) const noexcept;
        const auto data() const noexcept;

      protected:
        node_op(){};
        ~node_op() = default;

        node_op(const node_op &) = default;
        node_op &operator=(const node_op &) = default;

        node_op(node_op &&) = default;
        node_op &operator=(node_op &&) = default;
    };

    /**************************
     * node_op implementation *
     **************************/

    template<class D>
        inline auto node_op<D>::derived_cast() & noexcept -> derived_type &
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D>
        inline auto node_op<D>::derived_cast() const &
        noexcept -> const derived_type &
    {
        return *static_cast<const derived_type *>(this);
    }

    template<class D>
        inline auto node_op<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D>
    inline auto node_op<D>::index(int i) const noexcept
    {
        return this->derived_cast().m_data.index(i);
    }

    template<class D>
    inline auto node_op<D>::size(std::size_t dim) const noexcept
    {
        return this->derived_cast().m_data.size(dim);
    }

    template<class D>
    inline auto node_op<D>::start(std::size_t dim, std::size_t index) const
        noexcept
    {
        return this->derived_cast().m_data.start(dim, index);
    }

    template<class D>
    inline auto node_op<D>::end(std::size_t dim, std::size_t index) const
        noexcept
    {
        return this->derived_cast().m_data.start(dim, index);
    }

    template<class D>
    inline auto node_op<D>::offset(std::size_t dim, std::size_t index) const
        noexcept
    {
        return this->derived_cast().m_data.offset(dim, index);
    }

    template<class D>
    inline auto node_op<D>::interval(std::size_t dim, std::size_t index) const
        noexcept
    {
        return this->derived_cast().m_data.interval(dim, index);
    }

    template<class D>
    inline auto node_op<D>::offsets_size(std::size_t dim) const noexcept
    {
        return this->derived_cast().m_data.offsets_size(dim);
    }

    template<class D>
    inline const auto node_op<D>::data() const noexcept
    {
        return this->derived_cast().m_data.data();
    }

    template<class E>
    using is_node_op = xt::is_crtp_base_of<node_op, E>;

    /************************
     * mesh_node definition *
     ************************/

    template<class Mesh>
    struct mesh_node : public node_op<mesh_node<Mesh>>
    {
        using mesh_type = Mesh;
        static constexpr std::size_t dim = mesh_type::dim;
        static constexpr std::size_t level = mesh_type::level;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;

        mesh_node(const Mesh &v);
        mesh_node(const mesh_node &) = default;
        mesh_node &operator=(const mesh_node &) = default;

        mesh_node(mesh_node &&) = default;
        mesh_node &operator=(mesh_node &&) = default;

        auto index(int i) const noexcept;
        auto size(std::size_t dim) const noexcept;
        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;
        auto offset(std::size_t dim, std::size_t off_ind) const noexcept;
        auto offsets_size(std::size_t dim) const noexcept;
        auto interval(std::size_t dim, std::size_t index) const noexcept;
        const Mesh &data() const noexcept;

      private:
        const Mesh &m_data;
    };

    /****************************
     * mesh_node implementation *
     ****************************/

    template<class Mesh>
    inline mesh_node<Mesh>::mesh_node(const Mesh &v) : m_data{v}
    {}

    template<class Mesh>
    inline auto mesh_node<Mesh>::index(int i) const noexcept
    {
        return i;
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::size(std::size_t dim) const noexcept
    {
        return m_data[dim].size();
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::start(std::size_t dim, std::size_t index) const
        noexcept
    {
        return m_data[dim][index].start;
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::end(std::size_t dim, std::size_t index) const
        noexcept
    {
        return m_data[dim][index].end;
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::offset(std::size_t dim,
                                        std::size_t off_ind) const noexcept
    {
        return m_data.offsets(dim)[off_ind];
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::offsets_size(std::size_t dim) const noexcept
    {
        return m_data.offsets(dim).size();
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::interval(std::size_t dim,
                                          std::size_t index) const noexcept
    {
        return m_data[dim][index];
    }

    template<class Mesh>
    inline const Mesh &mesh_node<Mesh>::data() const noexcept
    {
        return m_data;
    }

    /***************************
     * translate_op definition *
     ***************************/

    template<class T>
    struct translate_op : public node_op<translate_op<T>>
    {
        using mesh_type = T;
        static constexpr std::size_t dim = mesh_type::dim;
        static constexpr std::size_t level = mesh_type::level;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;

        translate_op(T &&v);
        translate_op(const T &v);

        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;

      private:
        T m_data;

        friend class node_op<translate_op<T>>;
    };

    /*******************************
     * translate_op implementation *
     *******************************/

    template<class T>
    inline translate_op<T>::translate_op(T &&v) : m_data{std::forward<T>(v)}
    {}

    template<class T>
    inline translate_op<T>::translate_op(const T &v) : m_data{v}
    {}

    template<class T>
    inline auto translate_op<T>::start(std::size_t dim, std::size_t index) const
        noexcept
    {
        if (dim == 0)
            return m_data.start(dim, index) + 1;
        return m_data.start(dim, index);
    }

    template<class T>
    inline auto translate_op<T>::end(std::size_t dim, std::size_t index) const
        noexcept
    {
        if (dim == 0)
            return m_data.end(dim, index) + 1;
        return m_data.end(dim, index);
    }

    /****************************
     * projection_op definition *
     ****************************/

    template<std::size_t ref_level, class T>
    struct projection_op : public node_op<projection_op<ref_level, T>>
    {
        using mesh_type = T;
        static constexpr std::size_t dim = mesh_type::dim;
        static constexpr std::size_t level = mesh_type::level;
        static constexpr long int shift = level - ref_level;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;

        projection_op(T &&v);
        projection_op(const T &v);

        auto index(int i) noexcept;
        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;

      private:
        T m_data;

        friend class node_op<projection_op<ref_level, T>>;
    };

    /********************************
     * projection_op implementation *
     ********************************/

    template<std::size_t ref_level, class T>
    inline projection_op<ref_level, T>::projection_op(T &&v)
        : m_data{std::forward<T>(v)}
    {}

    template<std::size_t ref_level, class T>
    inline projection_op<ref_level, T>::projection_op(const T &v) : m_data{v}
    {}

    template<std::size_t ref_level, class T>
    inline auto projection_op<ref_level, T>::index(int i) noexcept
    {
        auto index_ = m_data.index(i);
        return (shift >= 0) ? index_ << shift : index_ >> -shift;
    }

    template<std::size_t ref_level, class T>
    inline auto projection_op<ref_level, T>::start(std::size_t dim,
                                                   std::size_t index) const
        noexcept
    {
        auto start_ = m_data.start(dim, index);
        return (shift >= 0) ? start_ >> shift : start_ << -shift;
    }

    template<std::size_t ref_level, class T>
    inline auto projection_op<ref_level, T>::end(std::size_t dim,
                                                 std::size_t index) const
        noexcept
    {
        auto end_ = m_data.end(dim, index);
        if (shift > 0 && end_ & 1)
            end_++;
        return (shift >= 0) ? end_ >> shift : end_ << -shift;
    }

    namespace detail
    {
        template<class T>
        struct get_arg_node_impl
        {
            template<class R>
            decltype(auto) operator()(R &&r)
            {
                return std::forward<R>(r);
            }
        };

        template<>
        template<std::size_t Dim, std::size_t Level, class TInterval>
        struct get_arg_node_impl<LevelCellArray<Dim, Level, TInterval>>
        {
            using mesh_t = LevelCellArray<Dim, Level, TInterval>;

            decltype(auto) operator()(LevelCellArray<Dim, Level, TInterval> &r)
            {
                return mesh_node<mesh_t>(r);
            }
        };
    }

    template<class T>
    decltype(auto) get_arg_node(T &&t)
    {
        detail::get_arg_node_impl<std::decay_t<T>> inv;
        return inv(std::forward<T>(t));
    }

    template<class T>
    inline auto translate(T &&t)
    {
        auto arg = get_arg_node(std::forward<T>(t));
        using arg_t = decltype(arg);
        return translate_op<arg_t>{std::forward<arg_t>(arg)};
    }

    template<std::size_t ref_level, class T>
    inline auto projection(T &&t)
    {
        auto arg = get_arg_node(std::forward<T>(t));
        using arg_t = decltype(arg);
        return projection_op<ref_level, arg_t>{std::forward<arg_t>(arg)};
    }
}