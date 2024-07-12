#
# CMake build configuration for mpi_matrix library
#
cmake_minimum_required(VERSION 3.20)

project(mpi_matrix
        VERSION 0.0.1
        DESCRIPTION "Matrix structures with MPI-distributed storage"
        LANGUAGES C
    )
include(GNUInstallDirs)

option(ENABLE_INT64 "Use 64-bit integers for indices et al." ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_PIC_STATIC_LIBS "Build static libraries with position-independent code" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Select C11 or C23 standard when available:
if ("c_std_23" IN_LIST CMAKE_C_COMPILE_FEATURES )
    set(CMAKE_C_STANDARD 23)
elseif ( "c_std_11" IN_LIST CMAKE_C_COMPILE_FEATURES )
    set(CMAKE_C_STANDARD 11)
endif ()

# We need MPI and threading:
find_package(MPI REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#
# Add project info/version variables for the sake of the configure file:
#
set(MPI_MATRIX_NAME ${PROJECT_NAME})
set(MPI_MATRIX_VERSION ${mpi_dist_matrix_VERSION})

#
# Generate our common header file and have all targets in this
# directory use the binary directory on the search path:
#
configure_file(mpi_matrix_config.h.in mpi_matrix_config.h)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#
# The library:
#
set(mpi_matrix_SOURCES
        mpi_utils.c
        int_set.c
        mpi_matrix_coord.c
        mpi_matrix_storage.c
        mpi_matrix_stream.c
        mpi_matrix_dist_coord.c)
        
set(mpi_matrix_HEADERS
        ${CMAKE_CURRENT_BINARY_DIR}/mpi_matrix_config.h
        int_pair.h
        int_range.h
        int_set.h
        mpi_utils.h
        mpi_matrix_coord.h
        mpi_matrix_storage.h
        mpi_matrix_dist_coord.h)

if ( ${BUILD_STATIC_LIBS} )
    add_library(mpi_matrix_static STATIC ${mpi_matrix_SOURCES})
    target_compile_options(mpi_matrix_static PRIVATE ${MPI_C_COMPILE_FLAGS})
    target_include_directories(mpi_matrix_static PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_directories(mpi_matrix_static PRIVATE ${MPI_C_LINK_FLAGS})
    target_link_libraries(mpi_matrix_static PRIVATE m Threads::Threads ${MPI_C_LIBRARIES})
    set_target_properties(mpi_matrix_static PROPERTIES
            LIBRARY_OUTPUT_NAME mpi_matrix
            POSITION_INDEPENDENT_CODE ${BUILD_PIC_STATIC_LIBS}
            PUBLIC_HEADER "${mpi_matrix_HEADERS}"
        )
    add_library(mpi_matrix_test_lib ALIAS mpi_matrix_static)
endif ()

if ( ${BUILD_SHARED_LIBS} )
    add_library(mpi_matrix_shared SHARED ${mpi_matrix_SOURCES})
    target_compile_options(mpi_matrix_shared PRIVATE ${MPI_C_COMPILE_FLAGS})
    target_include_directories(mpi_matrix_shared PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_directories(mpi_matrix_shared PRIVATE ${MPI_C_LINK_FLAGS})
    target_link_libraries(mpi_matrix_shared PRIVATE m Threads::Threads ${MPI_C_LIBRARIES})
    set_target_properties(mpi_matrix_shared PROPERTIES
            LIBRARY_OUTPUT_NAME mpi_matrix
            POSITION_INDEPENDENT_CODE ON
            PUBLIC_HEADER "${mpi_matrix_HEADERS}"
        )
    if (NOT TARGET mpi_matrix_test_lib)
        add_library(mpi_matrix_test_lib ALIAS mpi_matrix_shared)
    endif ()
endif ()

#
# Test programs:
#
add_executable(mpi_matrix_coord_test mpi_matrix_coord_test.c)
target_compile_options(mpi_matrix_coord_test PRIVATE ${MPI_C_COMPILE_FLAGS})
target_include_directories(mpi_matrix_coord_test PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_directories(mpi_matrix_coord_test PRIVATE ${MPI_C_LINK_FLAGS})
target_link_libraries(mpi_matrix_coord_test PRIVATE Threads::Threads ${MPI_C_LIBRARIES} mpi_matrix_test_lib)

add_executable(mpi_matrix_storage_test mpi_matrix_storage_test.c)
set_property(TARGET mpi_matrix_storage_test PROPERTY C_STANDARD 23)
target_compile_options(mpi_matrix_storage_test PRIVATE ${MPI_C_COMPILE_FLAGS})
target_include_directories(mpi_matrix_storage_test PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_directories(mpi_matrix_storage_test PRIVATE ${MPI_C_LINK_FLAGS})
target_link_libraries(mpi_matrix_storage_test PRIVATE Threads::Threads ${MPI_C_LIBRARIES} mpi_matrix_test_lib)

add_executable(mpi_matrix_storage_test_io mpi_matrix_storage_test_io.c)
set_property(TARGET mpi_matrix_storage_test_io PROPERTY C_STANDARD 23)
target_compile_options(mpi_matrix_storage_test_io PRIVATE ${MPI_C_COMPILE_FLAGS})
target_include_directories(mpi_matrix_storage_test_io PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_directories(mpi_matrix_storage_test_io PRIVATE ${MPI_C_LINK_FLAGS})
target_link_libraries(mpi_matrix_storage_test_io PRIVATE Threads::Threads ${MPI_C_LIBRARIES} mpi_matrix_test_lib)

add_executable(mpi_matrix_stream_test mpi_matrix_stream_test.c)
target_compile_options(mpi_matrix_stream_test PRIVATE ${MPI_C_COMPILE_FLAGS})
target_include_directories(mpi_matrix_stream_test PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_directories(mpi_matrix_stream_test PRIVATE ${MPI_C_LINK_FLAGS})
target_link_libraries(mpi_matrix_stream_test PRIVATE Threads::Threads ${MPI_C_LIBRARIES} mpi_matrix_test_lib)

add_executable(mpi_matrix_dist_coord_test mpi_matrix_dist_coord_test.c)
target_compile_options(mpi_matrix_dist_coord_test PRIVATE ${MPI_C_COMPILE_FLAGS})
target_include_directories(mpi_matrix_dist_coord_test PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_directories(mpi_matrix_dist_coord_test PRIVATE ${MPI_C_LINK_FLAGS})
target_link_libraries(mpi_matrix_dist_coord_test PRIVATE Threads::Threads ${MPI_C_LIBRARIES} mpi_matrix_test_lib)

#
# Install target(s):
#
if ( ${BUILD_SHARED_LIBS} )
    install(TARGETS mpi_matrix_shared PUBLIC_HEADER)
endif ()
if ( ${BUILD_STATIC_LIBS} )
    install(TARGETS mpi_matrix_static PUBLIC_HEADER)
endif ()
