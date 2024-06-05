
#include "mpi_matrix_storage.h"

int
main()
{
    mpi_matrix_coord_t      c1;
    mpi_matrix_storage_ptr  m1;
    struct timespec         t1, t2;
    
    clock_gettime(CLOCK_REALTIME, &t1);
    mpi_matrix_coord_init(&c1, mpi_matrix_coord_type_upper_triangular, false, int_pair_make(10000000, 10000000));
    m1 = mpi_matrix_storage_create(mpi_matrix_storage_type_sparse_bst, mpi_matrix_storage_datatype_real_dp, &c1);
    if ( m1 ) {
        int_pair_t          p;
        base_int_t          i;
        
        printf("m1 allocated with type [%s,%s] using %lu bytes and %s-major %s coordinates of dimension (" BASE_INT_FMT "," BASE_INT_FMT ")\n",
                mpi_matrix_storage_get_type_name(m1),
                mpi_matrix_storage_get_datatype_name(m1),
                mpi_matrix_storage_byte_usage(m1),
                m1->coord.is_row_major ? "row" : "column",
                mpi_matrix_coord_get_type_name(&m1->coord),
                m1->coord.dimensions.i,
                m1->coord.dimensions.j
            );
        
        for ( p.i=0; p.i < 10; p.i++ ) {
            for (p.j = 0; p.j < 10; p.j++ ) {
                double      v = sqrt((double)p.i * (double)p.i + (1.0 + (double)p.j) * (double)p.j);
                
                mpi_matrix_storage_set(m1, mpi_matrix_orient_normal, p, &v);
            }
        }
        
        mpi_matrix_storage_clear(m1, mpi_matrix_orient_normal, int_pair_make(9, 0));
        mpi_matrix_storage_clear(m1, mpi_matrix_orient_normal, int_pair_make(9, 2));
        
        for ( i=0; i < 10000; i++ ) {
            double          v;
            
            p.i = 10 + rand() % (c1.dimensions.i - 10);
            p.j = 10 + rand() % (c1.dimensions.j - 10);
            v = sin(p.i + p.j);
            mpi_matrix_storage_set(m1, mpi_matrix_orient_normal, p, &v);
        }
        
        clock_gettime(CLOCK_REALTIME, &t2);
        
        double      dt = (t2.tv_sec - t1.tv_sec) + 1e-9 * (t2.tv_nsec - t1.tv_nsec);
        
        printf("...after init of [0,0]-[9,9] sub-matrix, removal of 2 elements, and 1000 random additions requiring %.3lg s of wall time\n", dt);
        printf("m1 with type [%s,%s] using %lu bytes and %s-major %s coordinates of dimension (" BASE_INT_FMT "," BASE_INT_FMT ")\n",
                mpi_matrix_storage_get_type_name(m1),
                mpi_matrix_storage_get_datatype_name(m1),
                mpi_matrix_storage_byte_usage(m1),
                m1->coord.is_row_major ? "row" : "column",
                mpi_matrix_coord_get_type_name(&m1->coord),
                m1->coord.dimensions.i,
                m1->coord.dimensions.j
            );
        printf("\n");
        for ( p.i=0; p.i < 10; p.i++ ) {
            for (p.j = 0; p.j < 10; p.j++ ) {
                double      v;
                printf((mpi_matrix_storage_get(m1, mpi_matrix_orient_normal, p, &v) ? "%12.4lg" : "            "), v);
            }
            printf("\n");
        }
        
        mpi_matrix_storage_ptr  m2;
        
        clock_gettime(CLOCK_REALTIME, &t1);
        if ( mpi_matrix_storage_sparse_bst_to_compressed(m1, &m2) ) {
            base_int_t          nvalues;
            const base_int_t    *columns;
            const double        *values;
            
            clock_gettime(CLOCK_REALTIME, &t2);
            dt = (t2.tv_sec - t1.tv_sec) + 1e-9 * (t2.tv_nsec - t1.tv_nsec);
            printf("\nafter conversion to compressed sparse requiring %.3lg s\n", dt);
            printf("m2 allocated with type [%s,%s] using %lu bytes and %s-major %s coordinates of dimension (" BASE_INT_FMT "," BASE_INT_FMT ")\n",
                    mpi_matrix_storage_get_type_name(m2),
                    mpi_matrix_storage_get_datatype_name(m2),
                    mpi_matrix_storage_byte_usage(m2),
                    m2->coord.is_row_major ? "row" : "column",
                    mpi_matrix_coord_get_type_name(&m2->coord),
                    m2->coord.dimensions.i,
                    m2->coord.dimensions.j
                );
        
            if ( mpi_matrix_storage_sparse_compressed_get_fields(m2, NULL, NULL, &nvalues, NULL, &columns, (const void**)&values) ) {
                base_int_t      max_print = 20;
                
                printf("\nnvalues = " BASE_INT_FMT ":\n\n", nvalues);
                while ( nvalues-- && max_print-- )
                    printf(BASE_INT_FMT " = %lg\n", *columns++, *values++);
                printf(nvalues ? "  :\n\n" : "\n");
            }
            mpi_matrix_storage_destroy(m2);
        }
        mpi_matrix_storage_destroy(m1);
    }
    
    return 0;
}
