
#include "mpi_matrix_dist_coord.h"
#include "mpi_utils.h"

int
main(
    int             argc,
    const char*     argv[]
)
{
    mpi_matrix_dist_coord_ptr   dc1;
    mpi_matrix_coord_t          c1;
    int_pair_t                  dc1_hint;
    const char                  *err_msg;
    
    MPI_Init(&argc, (char***)&argv);
    
    mpi_matrix_coord_init(&c1, mpi_matrix_coord_type_full, true, int_pair_make(1000, 1000));
    
    mpi_printf(0, "Creating global coordinate system:\n");
    mpi_printf(0, "    mpi_matrix_coord@%p {%s, row-major=%c, dims=(" BASE_INT_FMT "," BASE_INT_FMT ")}\n", &c1, mpi_matrix_coord_get_type_name(&c1), c1.is_row_major ? 't' : 'f', c1.dimensions.i, c1.dimensions.j);
    
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_printf(0, "Creating distributed coordinate system with ratio hint (1,0) -- by-rows, all-columns:\n");
    fflush(stdout);
    dc1_hint = int_pair_make(1, 0);
    dc1 = mpi_matrix_dist_coord_create(
                mpi_matrix_dist_coord_split_by_rank,
                &c1,
                MPI_COMM_WORLD,
                mpi_matrix_dist_coord_hint_ratio,
                &dc1_hint,
                &err_msg);
    if ( dc1 ) {
        mpi_seq_printf("    mpi_matrix_dist_coord@%p {%s, base=(" BASE_INT_FMT "," BASE_INT_FMT "), dims=(" BASE_INT_FMT "," BASE_INT_FMT ")}\n", dc1, mpi_matrix_coord_get_type_name(&dc1->local_coord), dc1->local_index.i, dc1->local_index.j, dc1->local_coord.dimensions.i, dc1->local_coord.dimensions.j);
        mpi_matrix_dist_coord_destroy(dc1);
    } else {
        mpi_seq_printf("    mpi_matrix_dist_coord_create failed:  %s\n", err_msg);
    }
    
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_printf(0, "Creating distributed coordinate system with ratio hint (0,1) -- by-columns, all-rows:\n");
    dc1_hint = int_pair_make(0, 1);
    dc1 = mpi_matrix_dist_coord_create(
                mpi_matrix_dist_coord_split_by_rank,
                &c1,
                MPI_COMM_WORLD,
                mpi_matrix_dist_coord_hint_ratio,
                &dc1_hint,
                &err_msg);
    if ( dc1 ) {
        mpi_seq_printf("    mpi_matrix_dist_coord@%p {%s, base=(" BASE_INT_FMT "," BASE_INT_FMT "), dims=(" BASE_INT_FMT "," BASE_INT_FMT ")}\n", dc1, mpi_matrix_coord_get_type_name(&dc1->local_coord), dc1->local_index.i, dc1->local_index.j, dc1->local_coord.dimensions.i, dc1->local_coord.dimensions.j);
        mpi_matrix_dist_coord_destroy(dc1);
    } else {
        mpi_seq_printf("    mpi_matrix_dist_coord_create failed:  %s\n", err_msg);
    }
    
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_printf(0, "Creating distributed coordinate system with ratio hint (1,2) -- 2 columns per row:\n");
    fflush(stdout);
    dc1_hint = int_pair_make(1, 2);
    dc1 = mpi_matrix_dist_coord_create(
                mpi_matrix_dist_coord_split_by_rank,
                &c1,
                MPI_COMM_WORLD,
                mpi_matrix_dist_coord_hint_ratio,
                &dc1_hint,
                &err_msg);
    if ( dc1 ) {
        mpi_seq_printf("    mpi_matrix_dist_coord@%p {%s, base=(" BASE_INT_FMT "," BASE_INT_FMT "), dims=(" BASE_INT_FMT "," BASE_INT_FMT ")}\n", dc1, mpi_matrix_coord_get_type_name(&dc1->local_coord), dc1->local_index.i, dc1->local_index.j, dc1->local_coord.dimensions.i, dc1->local_coord.dimensions.j);
        mpi_matrix_dist_coord_destroy(dc1);
    } else {
        mpi_seq_printf("    mpi_matrix_dist_coord_create failed:  %s\n", err_msg);
    }
    
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_printf(0, "Creating distributed coordinate system with no hint -- 2d across all ranks:\n");
    fflush(stdout);
    dc1_hint = int_pair_make(1, 0);
    dc1 = mpi_matrix_dist_coord_create(
                mpi_matrix_dist_coord_split_by_rank,
                &c1,
                MPI_COMM_WORLD,
                mpi_matrix_dist_coord_hint_none,
                NULL,
                &err_msg);
    if ( dc1 ) {
        mpi_seq_printf("    mpi_matrix_dist_coord@%p {%s, base=(" BASE_INT_FMT "," BASE_INT_FMT "), dims=(" BASE_INT_FMT "," BASE_INT_FMT ")}\n", dc1, mpi_matrix_coord_get_type_name(&dc1->local_coord), dc1->local_index.i, dc1->local_index.j, dc1->local_coord.dimensions.i, dc1->local_coord.dimensions.j);
        mpi_matrix_dist_coord_destroy(dc1);
    } else {
        mpi_seq_printf("    mpi_matrix_dist_coord_create failed:  %s\n", err_msg);
    }
    
    fflush(stdout);
    
    mpi_printf(0,
            "Creating distributed coordinate system with no hint -- 2d by node (possible shared mem)\n");
    dc1_hint = int_pair_make(1, 0);
    dc1 = mpi_matrix_dist_coord_create(
                mpi_matrix_dist_coord_split_by_node,
                &c1,
                MPI_COMM_WORLD,
                mpi_matrix_dist_coord_hint_none,
                NULL,
                &err_msg);
    if ( dc1 ) {
        mpi_seq_printf("    mpi_matrix_dist_coord@%p {%s, base=(" BASE_INT_FMT "," BASE_INT_FMT "), dims=(" BASE_INT_FMT "," BASE_INT_FMT ")}\n", dc1, mpi_matrix_coord_get_type_name(&dc1->local_coord), dc1->local_index.i, dc1->local_index.j, dc1->local_coord.dimensions.i, dc1->local_coord.dimensions.j);
        mpi_matrix_dist_coord_destroy(dc1);
    } else {
        mpi_seq_printf("    mpi_matrix_dist_coord_create failed:  %s\n", err_msg);
    }
    
    MPI_Finalize();
    return 0;
}
