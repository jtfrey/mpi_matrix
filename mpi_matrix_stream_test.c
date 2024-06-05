
#include "mpi_matrix_stream.h"

#ifdef HAVE_BLAS
#include <blas.h>
#else

static inline double
ddot(
    base_int_t      n,
    const double    *dx,
    base_int_t      inc_dx,
    const double    *dy,
    base_int_t      inc_dy
)
{
    double          s = 0.0;
    
    while ( n-- ) {
        s += *dx * *dy;
        dx += inc_dx;
        dy += inc_dy;
    }
    return s;
}

static inline double
opt_ddot(
    base_int_t      n,
    const double    *dx,
    base_int_t      inc_dx,
    const double    *dy,
    base_int_t      inc_dy
)
{
    double          s;
    
    if ( n > 20)
        s = ddot(n, dx, inc_dx, dy, inc_dy);
    else {
        s = 0.0;
        while ( n-- ) {
            s += *dx * *dy;
            dx += inc_dx, dy += inc_dy;
        }
    }
    return s;
}

#endif

#ifndef N_ELEMENTS
#define N_ELEMENTS 10000
#endif

#ifndef N_ELEMENTS_ZERO
#define N_ELEMENTS_ZERO 8000
#endif

#ifndef FB_BYTE_CAPACITY
#define FB_BYTE_CAPACITY (32 * 1024)
#endif

double
randfrom(
    double  min,
    double  max
) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

base_int_t
randidx(
    base_int_t  i_max
)
{
    return rand() % i_max;
}

static inline base_int_t
base_int_abs(
    base_int_t  i
)
{
    return ( i < 0 ) ? -i : i;
}

int
mpi_printf(
    int         rank,
    const char  *fmt,
    ...
)
{
    static int my_rank, comm_size;
    static bool my_rank_inited = false;
    int     rc = 0;
    
    if ( ! my_rank_inited ) {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        my_rank_inited = true;
    }
    if ( (rank < 0) || (rank == my_rank) ) {
        va_list argv;
        
        va_start(argv, fmt);
        rc = printf("[%d/%d] ", my_rank, comm_size) + vprintf(fmt, argv) + printf("\n");
        va_end(argv);
    }
    return rc;
}

int
main(
    int         argc,
    char*       argv[]
)
{
    base_int_t                  n_buffer_sends = 0;
    int                         global_rank, global_size;
    mpi_matrix_framebuffer_ref  fb;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    srand (time( NULL));
    
    fb = mpi_matrix_framebuffer_create(mpi_matrix_storage_datatype_real_dp, FB_BYTE_CAPACITY);
    if ( fb ) {
        base_int_t              i = 0, i_rem, j;
        
        mpi_printf(-1, "Frame buffer allocated with capacity of %d bytes", FB_BYTE_CAPACITY);
        
        if ( global_rank == 0 ) {
            double      v[N_ELEMENTS];
            double      *local_v = v;
        
            while (i < N_ELEMENTS) v[i++] = randfrom(-1.0, +1.0);
            mpi_printf(-1, "Vector filled with %d random numbers [-1, +1] on all ranks", N_ELEMENTS);
        
            // Add a few random zeroes to the vector:
            for ( j = 0; j < N_ELEMENTS_ZERO; j++ ) v[randidx(N_ELEMENTS)] = 0.0;
            mpi_printf(-1, "Vector sprinkled with %.0f%% zeroes (%d of %d)", (float)(N_ELEMENTS_ZERO) * 100.0 / (float)N_ELEMENTS, N_ELEMENTS_ZERO, N_ELEMENTS);
            
            // Compile the vector into the frame buffer:
            while ( i > 0 ) {
                // Compile some more data into the fb:
                mpi_matrix_framebuffer_reset(fb);
                while ( i > 0 ) {
                    if ( fabs(*local_v) < 1e-8 ) {
                        if ( ! mpi_matrix_framebuffer_push(fb, 1, &i_rem) ) break;
                    } else {
                        if ( ! mpi_matrix_framebuffer_push(fb, 1, local_v, &i_rem) ) break;
                    }
                    if ( i_rem ) break;
                    local_v++;
                    i--;
                }
                mpi_printf(-1, "Frame buffer filled with " BASE_INT_FMT " frames, " BASE_INT_FMT " elements (" BASE_INT_FMT " remain).",
                        mpi_matrix_framebuffer_get_frame_count(fb),
                        mpi_matrix_framebuffer_get_element_count(fb),
                        i
                    );
                MPI_Send(
                    mpi_matrix_framebuffer_get_buffer_ptr(fb),
                    mpi_matrix_framebuffer_get_byte_capacity(fb),
                    MPI_BYTE,
                    1,
                    0,
                    MPI_COMM_WORLD);
                mpi_printf(-1, "|- Frame buffer sent to client rank");
                n_buffer_sends++;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            mpi_printf(-1, "Transfer of data completed, " BASE_INT_FMT " buffer sends", n_buffer_sends);
            mpi_printf(-1, "Self v.v=%lg", opt_ddot(N_ELEMENTS, v, 1, v, 1));
        } else {
            double  dot = 0.0;
            
            i = N_ELEMENTS;
            while ( i > 0 ) {
                const void  *iter = NULL;
                
                // Receive the compiled buffer to the processing buffer:
                MPI_Recv(
                    (void*)mpi_matrix_framebuffer_get_buffer_ptr(fb),
                    mpi_matrix_framebuffer_get_byte_capacity(fb),
                    MPI_BYTE,
                    0,
                    0,
                    MPI_COMM_WORLD,
                    NULL);
                mpi_printf(-1, "Recevied frame buffer from server rank");
                
                // Validate the processing buffer:
                if ( ! mpi_matrix_framebuffer_is_valid(fb) ) break;
                mpi_printf(-1, "|- Frame buffer is valid");
            
                // Iterate over the frame buffer:
                iter = NULL;
                mpi_printf(-1, "|- Iterating over frame buffer with " BASE_INT_FMT " frames", mpi_matrix_framebuffer_get_frame_count(fb));
                while ( (iter = mpi_matrix_framebuffer_iter_next_frame(fb, iter)) ) {
                    base_int_t      n_elements = mpi_matrix_framebuffer_iter_get_n_elements(fb, iter);
                
                    mpi_printf(-1, "    |- %s frame with " BASE_INT_FMT " elements",
                            (n_elements < 0) ? "Implicit" : "Explicit",
                            n_elements
                        );
                    if ( n_elements > 0 ) {
                        const double    *elements_ptr;
                    
                        mpi_matrix_framebuffer_iter_get_elements_ptr(fb, iter, (void**)&elements_ptr);
                        dot += opt_ddot(n_elements, elements_ptr, 1, elements_ptr, 1);
                    }
                    i -= base_int_abs(n_elements);
                }            
                // Reset for next iteration:
                mpi_matrix_framebuffer_reset(fb);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            mpi_printf(-1, "Receive of data completed, v.v=%lg", dot);
        }
    }
    MPI_Finalize();
    return 0;
}
