
#include "mpi_utils.h"

#include <float.h>

//

int
__mpi_printf_rank()
{
    static bool is_inited = false;
    static int rank;
    
    if ( ! is_inited ) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        is_inited = true;
    }
    return rank;
}

//

int
__mpi_printf_size()
{
    static bool is_inited = false;
    static int size;
    
    if ( ! is_inited ) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        is_inited = true;
    }
    return size;
}

//

int
__mpi_printf_digits()
{
    static bool is_inited = false;
    static int digits;
    
    if ( ! is_inited ) {
        int     size = __mpi_printf_size();
        digits = 1;
        while ( size >= 10 ) digits++, size /= 10;
        is_inited = true;
    }
    return digits;
}

//

int
mpi_vprintf(
    int         rank,
    const char  *fmt,
    va_list     argv
)
{
    static int  msg_id = 0;
    int         n = 0, fmt_len = strlen(fmt);

    if ( (rank == -1) || (rank == __mpi_printf_rank()) ) {
        int     digits = __mpi_printf_digits();
        
        n = printf("[%04x][MPI-%0*d:%0*d] ", msg_id, digits, __mpi_printf_rank(), digits, __mpi_printf_size());
        n += vprintf(fmt, argv);
        if ( (fmt_len == 0) || (*(fmt + strlen(fmt) - 1) != '\n') ) n += printf("\n");
    }
    msg_id++;
    return n;
}

//

int
mpi_printf(
    int         rank,
    const char  *fmt,
    ...
)
{
    int         n = 0, fmt_len = strlen(fmt);
    va_list     argv;

    va_start(argv, fmt);
    n = mpi_vprintf(rank, fmt, argv);
    va_end(argv);
    return n;
}

//

int
mpi_seq_printf(
    const char  *fmt,
    ...
)
{
    int         n, ball;
    va_list     argv;
    
    va_start(argv, fmt);
    if ( __mpi_printf_rank() == 0 ) {
        n = mpi_vprintf(__mpi_printf_rank(), fmt, argv);
        if ( __mpi_printf_size() > 1 ) {
            MPI_Send(&ball, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&ball, 1, MPI_INT, __mpi_printf_size() - 1, 0, MPI_COMM_WORLD, NULL);
        }
    } else {
        MPI_Recv(&ball, 1, MPI_INT, __mpi_printf_rank() - 1, 0, MPI_COMM_WORLD, NULL);
        n = mpi_vprintf(__mpi_printf_rank(), fmt, argv);
        MPI_Send(&ball, 1, MPI_INT, (__mpi_printf_rank() + 1) % __mpi_printf_size(), 0,MPI_COMM_WORLD);
    }
    va_end(argv);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    return n;
}

//

static base_int_t __mpi_auto_grid_2d_factors[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 0};

static inline base_int_t
__mpi_auto_grid_2d_dim_distance(
    base_int_t  r,
    base_int_t  c
)
{
    return ( r > c ) ? (r - c) : (c - r);
}

static inline bool
__mpi_auto_grid_2d_is_exact(
    base_int_t  *dim_global,
    base_int_t  r,
    base_int_t  c
)
{
    return ((dim_global[0] % r) == 0) && ((dim_global[1] % c) == 0);
}

bool
__mpi_auto_grid_2d(
    int         ranks,
    bool        must_be_exact,
    bool        is_verbose,
    base_int_t  *dim_global,
    base_int_t  *out_dim_blocks
)
{
    double      dim_ratio = (double)dim_global[0] / (double)dim_global[1];
    double      dim_ratio_distance;
    base_int_t  r, c, saved_r, saved_c, dim_distance;
    
    r = saved_r = ranks;
    c = saved_c = 1;
    dim_distance = __mpi_auto_grid_2d_dim_distance(r, c);
    dim_ratio_distance = fabs(dim_ratio - (double)ranks);
    
    // Special case if the rows == columns:
    if ( dim_global[0] == dim_global[1] ) {
        // Is the rank count a perfect square?
        double  whole_part, fractional_part;
        
        fractional_part = modf(sqrt((double)ranks), &whole_part);
        if ( fabs(fractional_part) < DBL_EPSILON ) {
            saved_r = saved_c = (base_int_t)whole_part;
            dim_distance = __mpi_auto_grid_2d_dim_distance(saved_r, saved_c);
            dim_ratio_distance = fabs(dim_ratio - 1.0);
            if ( ! must_be_exact || __mpi_auto_grid_2d_is_exact(dim_global, whole_part, whole_part) ) {
                out_dim_blocks[0] = out_dim_blocks[1] = (base_int_t)whole_part;
                if ( is_verbose ) mpi_printf(0, "perfect-square will work: exact=%d [" BASE_INT_FMT "," BASE_INT_FMT "]", must_be_exact, out_dim_blocks[0], out_dim_blocks[1]);
                return true;
            }
        }
    }
    // Factor the ranks count into integer products and find the
    // product whose row:column ratio is closes to the ratio of
    // the actual dimensions.
    while ( r > 1 ) {
        base_int_t      *factors = __mpi_auto_grid_2d_factors;
        
        while ( *factors ) {
            if ( (r % *factors) == 0 ) {
                // r is cleanly divisible by this factor:
                base_int_t  r2 = r / *factors, c2 = c * *factors;
                base_int_t  new_dim_distance = __mpi_auto_grid_2d_dim_distance(r2, c2);
                double      new_ratio_distance = fabs(dim_ratio - (double)r2 / (double)c2);
                
                if ( is_verbose ) mpi_printf(0, "testing: exact=%d [" BASE_INT_FMT "," BASE_INT_FMT "] |r-c|=" BASE_INT_FMT ", ∆(r/c)=%lf, is-exact=%d", must_be_exact, r2, c2, new_dim_distance, new_ratio_distance, __mpi_auto_grid_2d_is_exact(dim_global, r2, c2));
                // How close to the dimensional ratio?
                if ( (new_ratio_distance < dim_ratio_distance) && (new_dim_distance <= dim_distance) ) {
                    if ( ! must_be_exact || __mpi_auto_grid_2d_is_exact(dim_global, r2, c2) ) {
                        saved_r = r2;
                        saved_c = c2;
                        dim_distance = new_dim_distance;
                        dim_ratio_distance = new_ratio_distance;
                    }
                }
                // What if we flip the two values?
                new_ratio_distance = fabs(dim_ratio - (double)c2 / (double)r2);
                
                if ( is_verbose ) mpi_printf(0, "testing: exact=%d [" BASE_INT_FMT "," BASE_INT_FMT "] |r-c|=" BASE_INT_FMT ", ∆(r/c)=%lf, is-exact=%d", must_be_exact, c2, r2, new_dim_distance, new_ratio_distance, __mpi_auto_grid_2d_is_exact(dim_global, c2, r2));
                if ( (new_ratio_distance < dim_ratio_distance) && (new_dim_distance <= dim_distance) ) {
                    if ( ! must_be_exact || __mpi_auto_grid_2d_is_exact(dim_global, c2, r2) ) {
                        saved_r = c2;
                        saved_c = r2;
                        dim_distance = new_dim_distance;
                        dim_ratio_distance = new_ratio_distance;
                    }
                }
                r = r2;
                c = c2;
                break;
            }
            factors++;
        }
        // If we found no usable factor, we can't divide r any
        // further:
        if ( *factors == 0 ) break;
    }
    if ( must_be_exact && ! __mpi_auto_grid_2d_is_exact(dim_global, saved_r, saved_c) ) {
        if ( __mpi_auto_grid_2d(ranks, false, is_verbose, dim_global, out_dim_blocks) ) {
            // Which is better, ours or the inexact?
            double      alt_dim_ratio_distance = fabs(dim_ratio - (double)out_dim_blocks[0] / (double)out_dim_blocks[1]);
            
            if ( (dim_ratio_distance >= alt_dim_ratio_distance) || (dim_distance >= __mpi_auto_grid_2d_dim_distance(out_dim_blocks[0], out_dim_blocks[1])) ) return true;
        }
    }
    out_dim_blocks[0] = saved_r;
    out_dim_blocks[1] = saved_c;
    return true;
}

bool
mpi_auto_grid_2d(
    int         ranks,
    bool        must_be_exact,
    bool        is_verbose,
    base_int_t  *dim_global,
    base_int_t  *out_dim_blocks
)
{
    return ( __mpi_auto_grid_2d(ranks, true, is_verbose, dim_global, out_dim_blocks) && (! must_be_exact || __mpi_auto_grid_2d_is_exact(dim_global, out_dim_blocks[0], out_dim_blocks[1])) );
}
