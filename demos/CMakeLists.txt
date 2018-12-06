

add_executable(box box.cpp ${MURE_HEADERS})
target_link_libraries(box mure)

add_executable(interval_op interval_op.cpp ${MURE_HEADERS})
target_link_libraries(interval_op mure)

add_executable(mesh mesh.cpp ${MURE_HEADERS})
target_link_libraries(mesh mure)
