add_executable(weno-weno5 weno5.cpp)
target_link_libraries(weno-weno5 samurai)

add_executable(weno-weno5-amr weno5_amr.cpp)
target_link_libraries(weno-weno5-amr samurai)

add_executable(weno-heat-weno5-amr heat_weno5_amr.cpp)
target_link_libraries(weno-heat-weno5-amr samurai)

add_executable(weno-VF-level-set-houc5-amr VF_level_set_houc5_amr.cpp)
target_link_libraries(weno-VF-level-set-houc5-amr samurai)

add_executable(weno-VF-level-set-os5-amr VF_level_set_os5_amr.cpp)
target_link_libraries(weno-VF-level-set-os5-amr samurai)
