
find_package(GTest REQUIRED)
message("GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS}, GTEST_BOTH_LIBRARIES ${GTEST_BOTH_LIBRARIES}")
include_directories(${GTEST_INCLUDE_DIRS})

function(test_unit_add_executable NAME)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs)
    cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${NAME} ${__UNPARSED_ARGUMENTS})
    target_link_libraries(${NAME} PUBLIC kprocess ${GTEST_BOTH_LIBRARIES})
endfunction()

# unit test
test_unit_add_executable(test_warp_perspective test_warp_perspective.cpp)
test_unit_add_executable(test_connected_componets test_connected_componets.cpp)
test_unit_add_executable(test_pixel_aggregate test_pixel_aggregate.cpp)
test_unit_add_executable(test_masked_roi_align test_masked_roi_align.cpp)
