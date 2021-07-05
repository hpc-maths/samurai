find_package(cxxopts)

# add_executable(bench_laplacian_2d bench_laplacian_2d.cpp ${SAMURAI_HEADERS})
# target_link_libraries(bench_laplacian_2d samurai)

# add_executable(bench_laplacian_3d bench_laplacian_3d.cpp ${SAMURAI_HEADERS})
# target_link_libraries(bench_laplacian_3d samurai)


# add_executable(box box.cpp ${SAMURAI_HEADERS})
# target_link_libraries(box samurai)

# add_executable(interval_op interval_op.cpp ${SAMURAI_HEADERS})
# target_link_libraries(interval_op samurai)

# add_executable(mesh mesh.cpp ${SAMURAI_HEADERS})
# target_link_libraries(mesh samurai)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat")
set(CMAKE_CXX_STANDARD 14)

# add_executable(heaviside heaviside.cpp ${SAMURAI_HEADERS})
# target_link_libraries(heaviside samurai)

# add_executable(advection_1d advection_1d.cpp ${SAMURAI_HEADERS})
# target_link_libraries(advection_1d samurai)

# add_executable(test_1d test_1d.cpp ${SAMURAI_HEADERS})
# target_link_libraries(test_1d samurai)

# add_executable(lbm_d1q2_burgers LBM/D1Q2_Burgers.cpp ${SAMURAI_HEADERS})
# target_link_libraries(lbm_d1q2_burgers samurai)
# add_executable(lbm_d1q2_burgers LBM/D1Q2_Burgers.cpp ${SAMURAI_HEADERS})
# target_link_libraries(lbm_d1q2_burgers samurai)


# add_executable(adv_ana_2d LBM/test_analytic_advection_2d.cpp ${SAMURAI_HEADERS})
# target_link_libraries(adv_ana_2d samurai)

# add_executable(lbm_d1q2_foo LBM/D1Q2_foo.cpp ${SAMURAI_HEADERS})
# target_link_libraries(lbm_d1q2_foo samurai)

#add_executable(test_2d test_2d.cpp ${SAMURAI_HEADERS})
#target_link_libraries(test_2d samurai)

# add_executable(test_proj test_projection.cpp ${SAMURAI_HEADERS})
# target_link_libraries(test_proj samurai)

# add_executable(subset subset.cpp ${SAMURAI_HEADERS})
# target_link_libraries(subset samurai)

# add_executable(bench_level_cell_array bench_level_cell_array.cpp ${SAMURAI_HEADERS})
# target_link_libraries(bench_level_cell_array samurai)

# add_executable(operator 1D_operator.cpp ${SAMURAI_HEADERS})
# target_link_libraries(operator samurai)

# add_executable(operator operator.cpp ${SAMURAI_HEADERS})
# target_link_libraries(operator samurai)

# add_subdirectory(multiresolution)
add_subdirectory(LBM)
# add_subdirectory(FiniteVolume)
# add_subdirectory(FiniteVolume-MR)
# add_subdirectory(tutorial)
# add_subdirectory(p4est)
# add_subdirectory(pablo)
# add_subdirectory(apple)
