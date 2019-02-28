add_executable(bench_laplacian_2d bench_laplacian_2d.cpp ${MURE_HEADERS})
target_link_libraries(bench_laplacian_2d mure)

add_executable(bench_laplacian_3d bench_laplacian_3d.cpp ${MURE_HEADERS})
target_link_libraries(bench_laplacian_3d mure)


# add_executable(box box.cpp ${MURE_HEADERS})
# target_link_libraries(box mure)

# add_executable(interval_op interval_op.cpp ${MURE_HEADERS})
# target_link_libraries(interval_op mure)

# add_executable(mesh mesh.cpp ${MURE_HEADERS})
# target_link_libraries(mesh mure)

add_executable(heaviside heaviside.cpp ${MURE_HEADERS})
target_link_libraries(heaviside mure)

# add_executable(bench_level_cell_array bench_level_cell_array.cpp ${MURE_HEADERS})
# target_link_libraries(bench_level_cell_array mure)

# add_executable(operator 1D_operator.cpp ${MURE_HEADERS})
# target_link_libraries(operator mure)
