TP Burgers
==========================================



Theoretical framework
########

Problem and exact solution
**********************

We want to approximate the solution :math:`u: [0, T] \times \mathbb{R} \to \mathbb{R}` of the well-known 1D Burgers equation

.. math::
    \begin{cases}
        \partial_t u(t, x) + \partial_x ( \varphi(u)(t, x) ) = 0, \qquad t \in [0, T], \quad &x \in \mathbb{R}, \\
        u(t=0, x) = u_0(x), \qquad  &x \in \mathbb{R},
    \end{cases}

with flux :math:`\varphi (u) = u^2/2` and hat-like initial datum given by


.. math::
    u_0(x) = (1+x) \chi_{[-1, 0]}(x) + (1-x) \chi_{[0, 1]}(x).


Using the method of the characteristic with the Rankine-Hugoniot jump relation, it can be shown that the solution is given by

.. math::
    u(t, x) = \frac{1+x}{1+t} \chi_{[-1, t]}(x) + \frac{1-x}{1-t} \chi_{[t, 1]}(x),


so that we can say that the solution blows up at time :math:`T^{\star}` in the sense that

.. math::
    \begin{cases}
        u(t, \cdot) \in C^0 (\mathbb{R}) \cap &L^{\infty}(\mathbb{R}), \qquad t \in [0, T^{\star}), \\
        u(t, \cdot) \in &L^{\infty}(\mathbb{R}), \qquad t \in [T^{\star}, T]. \\
    \end{cases}


Discretization
**********************

We consider to work on a bounded domain :math:`[a, b]`, with :math:`2^{\overline{J}}` cells of size :math:`\Delta x = (b-a)/2^{\overline{J}}` given by

.. math::
    C_{k} = [x_{k-1/2}, x_{k+1/2}], \qquad x_{k-1/2} = a + (b-a) k \Delta x.




Time is discretized by considering :math:`N \in \mathbb{N}` time steps of step :math:`\Delta t` so that :math:`t^n = n\Delta t` for :math:`n = 0, \dots, N-1`.
We assume that the time step has been selected to fulfill the CFL condition

.. math::
    \Delta t \leq \frac{\Delta x}{\sup_{x \in \mathbb{R}}{|\varphi'(u_0(x))|}}.


We consider that

.. math::
    \overline{u}_{k}^n \simeq \frac{1}{\Delta x} \int_{x_k - \Delta x/2}^{x_k + \Delta x/2} u(t^n, x) \text{d}x.

The numerical Finite Volumes scheme comes under the form

.. math::
    \overline{u}^{n+1}_k = \overline{u}^{n}_k + \frac{\Delta t}{\Delta x} (F_{k - 1/2}^n - F_{k+1/2}^n),

where we utilize the upwind fluxes given by

.. math::
    F_{k - 1/2}^n = \mathcal{F}(\overline{u}^{n}_{k-1}, \overline{u}^{n}_k), \qquad \text{with} \quad
     \mathcal{F}(\overline{u}_L, \overline{u}_R) = \begin{cases}
                                                        \varphi(\overline{u}_L), \qquad \text{if} \quad \frac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} &\geq 0, \\
                                                        \varphi(\overline{u}_R), \qquad \text{if} \quad \frac{\varphi'(\overline{u}_L) + \varphi'(\overline{u}_R)}{2} &< 0.
                                                  \end{cases}

Another possible choice for the flux is given by the Lax-Friedrichs, which is generally more diffusive than the upwind flux

.. math::
    \mathcal{F}(\overline{u}_L, \overline{u}_R) = \frac{1}{2} (\varphi(\overline{u}_L) + \varphi(\overline{u}_R)) - \frac{\Delta t}{2\Delta x} (\overline{u}_R - \overline{u}_L).



Mesh adaptation
**********************

To perform the AMR adaptation, we employ the following criterion

.. math::
    \text{Split }C_{j, k} \quad \text{if} \quad |\partial_x \overline{u}_{j, k}| > \delta,

where the derivative on the cell is estimated with the following centered formula

.. math::
    \partial_x \overline{u}_{j, k} \simeq \frac{\overline{u}_{j, k + 1} - \overline{u}_{j, k - 1}}{2\Delta x_j}



Implementation on a uniform mesh
########

Constructing the mesh
**********************

We first have to specify how SAMURAI has to build the computational mesh.
We construct the labels for the different categories of cells, namely

.. code-block:: c++

    enum class SimpleID
    {
        cells = 0, // Leaves (where the computation is done)
        cells_and_ghosts = 1, // Leaves + ghosts
        count = 2, // Total number of cells categories
        reference = cells_and_ghosts // Which is the largest class including all the others
    };


and then we specify what are the features of the mesh, namely its spatial dimension, the number of allowed levels, the number of ghosts, the basic brick (namely intervals based on integers) and which are the keys for the different cell categories

.. code-block:: c++

    struct AMRConfig
    {
        static constexpr std::size_t dim = 1; // Spatial dimension
        static constexpr std::size_t max_refinement_level = 20; // Maximum allowed levels (when doing AMR)
        static constexpr std::size_t ghost_width = 1; // Number of ghosts on each side

        using interval_t = samurai::Interval<int>;
        using mesh_id_t = SimpleID;
    };

This being done, we have to practically specify how to construct such a mesh.
Our class inherits from a built-in class of SAMURAI called Mesh_base and we shall provide AMRConfig as template parameter


.. code-block:: c++

    template <class Config>
    class AMRMesh: public samurai::Mesh_base<AMRMesh<Config>, Config>
    {
    public:
        // Importing all the types used in what follows
        using base_type = samurai::Mesh_base<AMRMesh<Config>, Config>;
        using config = typename base_type::config;
        static constexpr std::size_t dim = config::dim;

        using mesh_id_t = typename base_type::mesh_id_t;
        using cl_type = typename base_type::cl_type;
        using lcl_type = typename base_type::lcl_type;


        // Constructors and related operators
        AMRMesh(const AMRMesh&) = default;
        AMRMesh& operator=(const AMRMesh&) = default;

        AMRMesh(AMRMesh&&) = default;
        AMRMesh& operator=(AMRMesh&&) = default;

        // Constructor starting from a cell list
        inline AMRMesh(const cl_type &cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
        {}

        // Constructor from a given box (domain)
        inline AMRMesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
        : base_type(b, start_level, min_level, max_level)
        {}

        // This specifies how to add the ghosts once we know the leaves
        void update_sub_mesh_impl()
        {
            cl_type cl;
            for_each_interval(this->m_cells[mesh_id_t::cells], [&](std::size_t level, const auto& interval, auto)
            {
                lcl_type& lcl = cl[level];
                samurai::static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>([&](auto stencil)
                {
                    // We add as much ghosts in the given direction
                    // as prescribed by ghost_width
                    lcl[{}].add_interval({interval.start - config::ghost_width,
                                          interval.end   + config::ghost_width});
                });
            });
            // Put into the cells_and_ghosts category
            this->m_cells[mesh_id_t::cells_and_ghosts] = {cl, false};
        }
    };


We are ready to construct the mesh in the main function of our code


.. code-block:: c++

    int main(int argc, char *argv[])
    {
        using Config = AMRConfig;

        std::size_t max_level = 8;
        std::size_t min_level = max_level; // We have a uniform mesh

        samurai::Box<double, dim> box({-3}, {3}); // The domain is [-3, 3]
        AMRMesh<Config> mesh{box, max_level, min_level, max_level};

        // ...

    }


Initialize the solution on the mesh
**********************

We initialize the solution to the hat-like shape we have seen before.
For the sake of simplicity, we take the values at the cell centers

.. code-block:: c++

    // This construct a scalar field of doubles called
    // "phi" on the mesh
    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    using mesh_id_t = typename AMRMesh<Config>::mesh_id_t;

    // We loop on each cell of the mesh
    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center(); // Gets the cell center
        double x = center[0]; // Gets the first coordinate

        if (x < 1. or x > 1.)   {
            phi[cell] = 0.;
        }
        else    {
            phi[cell] = (x < 0.) ? (1 + x) : (1 - x);
        }
    });



Defining the numerical scheme
**********************

Now we have to define the flux of the numerical scheme.
We employ inheritance from "field_operator_base" which wants to have a "left_flux" and a "right_flux" and then merges it to obtain the Finite Volumes scheme


.. code-block:: c++

    template<class TInterval>
    class upwind_Burgers_op : public field_operator_base<TInterval>,
                            public finite_volume<upwind_Burgers_op<TInterval>> {
        public:
        INIT_OPERATOR(upwind_Burgers_op)

        template<class T1, class T2>
        inline auto flux(T1&& ul, T2&& ur) const
        {
            return xt::eval(0.5 * xt::pow(std::forward<T1>(ul), 2.)); // Upwind (works for positive solutions)
        }

        template<class T1>
        inline auto left_flux(const T1 &u) const
        {
            return flux(u(level, i-1), u(level, i));
        }

        template<class T1>
        inline auto right_flux(const T1 &u) const
        {
            return flux(u(level, i), u(level, i+1));
        }
    };

    // Wraps up the operator
    template<class... CT>
    inline auto upwind_Burgers(CT &&... e)
    {
        return make_field_operator_function<upwind_Burgers_op>(std::forward<CT>(e)...);
    }


Time stepping
**********************


.. code-block:: c++

    double Tf = 1.5; // In order to observe the blowup at t = 1
    double dx = 1./(1 << max_level);
    double dt = 0.99 * dx; // We work at CFL = 1/0.99

    double t = 0.;
    std::size_t it = 0;

    while (t < Tf)  {
        // New solution
        auto phinp1 = samurai::make_field<double, 1>("phi", mesh);
        phinp1 = phi - dt * samurai::upwind_Burgers(phi);
        std::swap(phi.array(), phinp1.array());

        t  += dt;
        it += 1;
    }
