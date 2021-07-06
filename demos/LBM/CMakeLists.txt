add_executable(high_order_mr high_order_mr.cpp)
target_link_libraries(high_order_mr samurai)

# 1D Paper - Verification of the error estimates

add_executable(test_D1Q2_high_order test_D1Q2_high_order.cpp)
target_link_libraries(test_D1Q2_high_order samurai)

# add_executable(test_D1Q2 test_D1Q2.cpp)
# target_link_libraries(test_D1Q2 samurai)

# add_executable(test_D1Q3 test_D1Q3.cpp)
# target_link_libraries(test_D1Q3 samurai)

# add_executable(test_D1Q5 test_D1Q5.cpp)
# target_link_libraries(test_D1Q5 samurai)

# add_executable(test_D1Q222 test_D1Q222.cpp)
# target_link_libraries(test_D1Q222 samurai)

# 1D Paper - Plot solutions

# add_executable(D1Q2_Advection_and_Burgers D1Q2_Advection_and_Burgers.cpp)
# target_link_libraries(D1Q2_Advection_and_Burgers samurai)

# add_executable(D1Q3_Shallow_Waters_Dam D1Q3_Shallow_Waters_Dam.cpp)
# target_link_libraries(D1Q3_Shallow_Waters_Dam samurai)

# add_executable(D1Q5_Shallow_Waters_Dam D1Q5_Shallow_Waters_Dam.cpp)
# target_link_libraries(D1Q5_Shallow_Waters_Dam samurai)

# add_executable(D1Q222_Euler_Sod D1Q222_Euler_Sod.cpp)
# target_link_libraries(D1Q222_Euler_Sod samurai)


# 2D Paper

# add_executable(D2Q4444_Euler_Lax_Liu D2Q4444_Euler_Lax_Liu.cpp)
# target_link_libraries(D2Q4444_Euler_Lax_Liu samurai)

add_executable(D2Q4444_Euler_Lax_Liu_uniform D2Q4444_Euler_Lax_Liu_uniform.cpp)
target_link_libraries(D2Q4444_Euler_Lax_Liu_uniform samurai)

add_executable(D2Q9_Navier_Stokes_von_Karman_street D2Q9_Navier_Stokes_von_Karman_street.cpp)
target_link_libraries(D2Q9_Navier_Stokes_von_Karman_street samurai)

# Additional 2D tests

# add_executable(D2Q4444_Euler_Implosion D2Q4444_Euler_Implosion.cpp)
# target_link_libraries(D2Q4444_Euler_Implosion samurai)

# add_executable(D2Q5444_Euler_Rayleigh_Taylor D2Q5444_Euler_Rayleigh_Taylor.cpp)
# target_link_libraries(D2Q5444_Euler_Rayleigh_Taylor samurai)
