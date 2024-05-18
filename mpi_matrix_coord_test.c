
#include "mpi_matrix_coord.h"

int
main()
{
    mpi_matrix_coord_t      mat_coord;
    int_pair_t              mat_dims = int_pair_make(1000, 500);
    int_pair_t              p;
    bool                    did_try_again = false;
    
try_again:

    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_full, true, mat_dims);
    printf("initialized     :  (" BASE_INT_FMT "," BASE_INT_FMT ")\n", mat_dims.i, mat_dims.j);
    
    p = int_pair_make(500, 250);
    printf("%-16s:  (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT " with allocation size " BASE_INT_FMT "\n",
        mpi_matrix_coord_get_type_name(&mat_coord), p.i, p.j, mpi_matrix_coord_index_to_offset(&mat_coord, false, p), mpi_matrix_coord_element_count(&mat_coord));
    
    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_upper_triangular, true, mat_dims);
    printf("%-16s:  (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT " with allocation size " BASE_INT_FMT "\n",
        mpi_matrix_coord_get_type_name(&mat_coord), p.i, p.j, mpi_matrix_coord_index_to_offset(&mat_coord, false, p), mpi_matrix_coord_element_count(&mat_coord));
    
    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_lower_triangular, true, mat_dims);
    printf("%-16s:  (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT " with allocation size " BASE_INT_FMT "\n",
        mpi_matrix_coord_get_type_name(&mat_coord), p.i, p.j, mpi_matrix_coord_index_to_offset(&mat_coord, false, p), mpi_matrix_coord_element_count(&mat_coord));
    
    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_diagonal, true, mat_dims);
    printf("%-16s:  (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT " with allocation size " BASE_INT_FMT "\n",
        mpi_matrix_coord_get_type_name(&mat_coord), p.i, p.j, mpi_matrix_coord_index_to_offset(&mat_coord, false, p), mpi_matrix_coord_element_count(&mat_coord));
    
    if ( mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_band_diagonal, true, mat_dims, 10, 12) ) {
        int_pair_t      band_coord = int_pair_make(234, 230);
        printf("%-16s:  (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT " with allocation size " BASE_INT_FMT " and k1=" BASE_INT_FMT ", k2=" BASE_INT_FMT "\n",
            mpi_matrix_coord_get_type_name(&mat_coord), band_coord.i, band_coord.j, mpi_matrix_coord_index_to_offset(&mat_coord, false, band_coord), mpi_matrix_coord_element_count(&mat_coord), mat_coord.k1, mat_coord.k2);
    }
    
    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_tridiagonal, true, mat_dims);
    printf("%-16s:  (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT " with allocation size " BASE_INT_FMT "\n",
        mpi_matrix_coord_get_type_name(&mat_coord), p.i, p.j, mpi_matrix_coord_index_to_offset(&mat_coord, false, p), mpi_matrix_coord_element_count(&mat_coord));
    printf("\n");
    
    if ( ! did_try_again ) {
        mat_dims = int_pair_make(1000, 1000);
        did_try_again = true;
        goto try_again;
    }
    
    printf("valid indices for row-major tridiagonal form:\n");
    for ( p.i = 0; p.i < 5; p.i++ ) {
        for ( p.j = 0; p.j < 6; p.j++ ) {
            base_int_t  o = mpi_matrix_coord_index_to_offset(&mat_coord, false, p);
            
            if ( o >= 0 ) printf("    (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT "\n", p.i, p.j, o);
        }
    }
    printf("\n");
    
    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_tridiagonal, false, mat_dims);
    printf("valid indices for column-major tridiagonal form:\n");
    for ( p.j = 0; p.j < 5; p.j++ ) {
        for ( p.i = 0; p.i < 6; p.i++ ) {
            base_int_t  o = mpi_matrix_coord_index_to_offset(&mat_coord, false, p);
            
            if ( o >= 0 ) printf("    (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT "\n", p.i, p.j, o);
        }
    }
    printf("\n");
    
    mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_diagonal, false, mat_dims);
    printf("valid indices for diagonal form:\n");
    for ( p.i = 0; p.i < 5; p.i++ ) {
        for ( p.j = 0; p.j < 5; p.j++ ) {
            base_int_t  o = mpi_matrix_coord_index_to_offset(&mat_coord, false, p);
            
            if ( o >= 0 ) printf("    (" BASE_INT_FMT "," BASE_INT_FMT ") = " BASE_INT_FMT "\n", p.i, p.j, o);
        }
    }
    printf("\n");
    
    if ( mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_band_diagonal, true, int_pair_make(10, 10), 5, 4) ) {
        printf("offset for indices, band-diagonal form (row-major, " BASE_INT_FMT " elements, k1=" BASE_INT_FMT ", k2=" BASE_INT_FMT "):\n", mpi_matrix_coord_element_count(&mat_coord), mat_coord.k1, mat_coord.k2);
        for ( p.i = 0; p.i < 10; p.i++ ) {
            for ( p.j = 0; p.j < 10; p.j++ ) {
                base_int_t  o = mpi_matrix_coord_index_to_offset(&mat_coord, false, p);
            
            
                if ( o >= 0 ) printf("%4" BASE_INT_FMT_NO_PCT, o);
                else printf("    ");
            }
            printf("\n");
        }
        printf("\n");
    }
    
    if ( mpi_matrix_coord_init(&mat_coord, mpi_matrix_coord_type_band_diagonal, false, int_pair_make(10, 10), 5, 4) ) {
        printf("offset for indices, band-diagonal form (column-major, " BASE_INT_FMT " elements, k1=" BASE_INT_FMT ", k2=" BASE_INT_FMT "):\n", mpi_matrix_coord_element_count(&mat_coord), mat_coord.k1, mat_coord.k2);
        for ( p.i = 0; p.i < 10; p.i++ ) {
            for ( p.j = 0; p.j < 10; p.j++ ) {
                base_int_t  o = mpi_matrix_coord_index_to_offset(&mat_coord, false, p);
            
            
                if ( o >= 0 ) printf("%4" BASE_INT_FMT_NO_PCT, o);
                else printf("    ");
            }
            printf("\n");
        }
        printf("\n");
    }
    
    return 0;
}
