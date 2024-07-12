
#ifndef __MPI_MATRIX_DIST_COORD_H__
#define __MPI_MATRIX_DIST_COORD_H__

#include "int_pair.h"
#include "mpi_matrix_coord.h"


typedef struct mpi_matrix_dist_coord * mpi_matrix_dist_coord_ptr;


typedef mpi_matrix_coord_status_t (*mpi_matrix_dist_coord_index_status_callback)(mpi_matrix_dist_coord_ptr coord, mpi_matrix_orient_t orient, int_pair_t p);

typedef bool (*mpi_matrix_dist_coord_index_reduce_callback)(mpi_matrix_dist_coord_ptr coord, mpi_matrix_orient_t orient, int_pair_t *p);

typedef int (*mpi_matrix_dist_coord_index_to_rank_callback)(mpi_matrix_dist_coord_ptr coord, mpi_matrix_orient_t orient, int_pair_t p);

typedef struct {
    mpi_matrix_dist_coord_index_status_callback     index_status;
    mpi_matrix_dist_coord_index_reduce_callback     index_reduce;
    mpi_matrix_dist_coord_index_to_rank_callback    index_to_rank;
} mpi_matrix_dist_coord_callbacks_t;



typedef enum {
    mpi_matrix_dist_coord_split_by_rank = 0,
    mpi_matrix_dist_coord_split_by_node,
#ifdef OPEN_MPI
    mpi_matrix_dist_coord_split_by_socket,
    mpi_matrix_dist_coord_split_by_numa,
#endif
    mpi_matrix_dist_coord_split_max
} mpi_matrix_dist_coord_split_t;


typedef enum {
    mpi_matrix_dist_coord_hint_none = 0,
    mpi_matrix_dist_coord_hint_exact,
    mpi_matrix_dist_coord_hint_ratio,
    mpi_matrix_dist_coord_hint_max
} mpi_matrix_dist_coord_hint_t;


typedef struct mpi_matrix_dist_coord {
    /*
     * Coordinate system of the full matrix:
     */
    mpi_matrix_coord_t                  global_coord;
    
    /*
     * MPI infrastructure:
     */
    MPI_Comm                            global_comm;
    int                                 global_rank, global_size;
    MPI_Comm                            dist_root_comm;
    int                                 *dist_root_ranks;
    int                                 dist_root_size;
    MPI_Comm                            local_comm;
    int                                 local_rank, local_size;
    
    /*
     * The local portion of the full matrix, as an initial
     * index and coordinate system relative to that index:
     */
    int                                 dist_root_idx;
    int_pair_t                          local_index;
    mpi_matrix_coord_t                  local_coord;
    
    mpi_matrix_dist_coord_callbacks_t   callbacks;
} mpi_matrix_dist_coord_t;



mpi_matrix_dist_coord_ptr
mpi_matrix_dist_coord_create(
    mpi_matrix_dist_coord_split_t   kind_of_split,
    mpi_matrix_coord_t              *global_coord,
    MPI_Comm                        global_comm,
    mpi_matrix_dist_coord_hint_t    kind_of_hint,
    int_pair_t                      *dist_hint,
    const char*                     *error_msg);

void mpi_matrix_dist_coord_destroy(mpi_matrix_dist_coord_ptr dist_coord);

#endif /* __MPI_MATRIX_DIST_COORD_H__ */
