# add_executable(bench_laplacian_2d bench_laplacian_2d.cpp ${MURE_HEADERS})
# target_link_libraries(bench_laplacian_2d mure)

# add_executable(bench_laplacian_3d bench_laplacian_3d.cpp ${MURE_HEADERS})
# target_link_libraries(bench_laplacian_3d mure)


# add_executable(box box.cpp ${MURE_HEADERS})
# target_link_libraries(box mure)

# add_executable(interval_op interval_op.cpp ${MURE_HEADERS})
# target_link_libraries(interval_op mure)

# add_executable(mesh mesh.cpp ${MURE_HEADERS})
# target_link_libraries(mesh mure)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat")
set(CMAKE_CXX_STANDARD 14)

# add_executable(heaviside heaviside.cpp ${MURE_HEADERS})
# target_link_libraries(heaviside mure)

# add_executable(advection_1d advection_1d.cpp ${MURE_HEADERS})
# target_link_libraries(advection_1d mure)

# add_executable(test_1d test_1d.cpp ${MURE_HEADERS})
# target_link_libraries(test_1d mure)

add_executable(lbm_d1q2 LBM/D1Q2.cpp ${MURE_HEADERS})
target_link_libraries(lbm_d1q2 mure)

#add_executable(test_2d test_2d.cpp ${MURE_HEADERS})
#target_link_libraries(test_2d mure)

# add_executable(test_proj test_projection.cpp ${MURE_HEADERS})
# target_link_libraries(test_proj mure)

# add_executable(subset subset.cpp ${MURE_HEADERS})
# target_link_libraries(subset mure)

# add_executable(bench_level_cell_array bench_level_cell_array.cpp ${MURE_HEADERS})
# target_link_libraries(bench_level_cell_array mure)

# add_executable(operator 1D_operator.cpp ${MURE_HEADERS})
# target_link_libraries(operator mure)

# add_executable(operator operator.cpp ${MURE_HEADERS})
# target_link_libraries(operator mure)
