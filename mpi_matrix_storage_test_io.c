
#include "mpi_matrix_storage.h"
#include <fcntl.h>
#include <sys/stat.h>

int
main()
{
    mpi_matrix_coord_t      c1;
    mpi_matrix_storage_ptr  m1;
    struct timespec         t1, t2;
    bool                    does_exist_on_disk = false;
    struct stat             finfo;
       
    clock_gettime(CLOCK_REALTIME, &t1);
    if ( stat("sparse-matrix.bin", &finfo) == 0 ) {
        int                 fd = open("sparse-matrix.bin", O_RDONLY);
        const char          *msg;
        
        if ( fd >= 0 ) {
            m1 = mpi_matrix_storage_read_from_fd(fd, true, &msg);
            close(fd);
            if ( ! m1 ) {
                printf("Failed to init m1 from file:  %s\n", msg);
            } else {
                int_pair_t          p;
                base_int_t          i;
                
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
                mpi_matrix_storage_destroy(m1);
            }
        }
    } else {
        mpi_matrix_coord_init(&c1, mpi_matrix_coord_type_upper_triangular, false, int_pair_make(10000000, 10000000));
        m1 = mpi_matrix_storage_create(mpi_matrix_storage_type_sparse_coordinate, mpi_matrix_storage_datatype_real_dp, &c1);
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
        
            int     fd = open("sparse-matrix.bin", O_CREAT|O_RDWR|O_TRUNC, 0777);
        
            if ( fd >= 0 ) {
                mpi_matrix_storage_write_to_fd(m1, fd, true);
            }
            mpi_matrix_storage_destroy(m1);
        }
    }

   clock_gettime(CLOCK_REALTIME, &t1);
    if ( stat("basic-matrix.bin", &finfo) == 0 ) {
        int                 fd = open("basic-matrix.bin", O_RDONLY);
        const char          *msg;
        
        if ( fd >= 0 ) {
            m1 = mpi_matrix_storage_read_from_fd(fd, true, &msg);
            close(fd);
            if ( ! m1 ) {
                printf("Failed to init m1 from file:  %s\n", msg);
            } else {
                int_pair_t          p;
                base_int_t          i;
                
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
                mpi_matrix_storage_destroy(m1);
            }
        }
    } else {
        mpi_matrix_coord_init(&c1, mpi_matrix_coord_type_upper_triangular, false, int_pair_make(100, 100));
        m1 = mpi_matrix_storage_create(mpi_matrix_storage_type_basic, mpi_matrix_storage_datatype_real_dp, &c1);
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
        
            for ( i=0; i < 1000; i++ ) {
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
        
            int     fd = open("basic-matrix.bin", O_CREAT|O_RDWR|O_TRUNC, 0777);
        
            if ( fd >= 0 ) {
                mpi_matrix_storage_write_to_fd(m1, fd, true);
            }
            mpi_matrix_storage_destroy(m1);
        }
    }



    return 0;
}
