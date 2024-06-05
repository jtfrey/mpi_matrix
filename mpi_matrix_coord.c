
#include "mpi_matrix_coord.h"

const char*
mpi_matrix_coord_type_get_name(
    mpi_matrix_coord_type_t     type
)
{
    static const char*  type_names[] = {
            "full",
            "upper-triangular",
            "lower-triangular",
            "band-diagonal",
            "tridiagonal",
            "diagonal",
            NULL
        };
    if ( type >= 0 && type < mpi_matrix_coord_type_max ) return type_names[type];
    return NULL;
}

//
////
//

base_int_t
__mpi_matrix_coord_full_element_count(
    mpi_matrix_coord_ptr    coord
)
{
    return coord->dimensions.i * coord->dimensions.j;
}

mpi_matrix_coord_status_t
__mpi_matrix_coord_full_index_status(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    return mpi_matrix_coord_status_is_defined | mpi_matrix_coord_status_is_unique;
}

bool
__mpi_matrix_coord_full_index_reduce(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              *p
)
{
    return true;
}

base_int_t
__mpi_matrix_coord_full_index_to_offset(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    if ( orient != mpi_matrix_orient_normal )
        return (coord->is_row_major) ? (p.j * coord->dimensions.j) + p.i :
                                       (p.i * coord->dimensions.i) + p.j;
    return (coord->is_row_major) ? (p.i * coord->dimensions.j) + p.j :
                                   (p.j * coord->dimensions.i) + p.i;
}

mpi_matrix_coord_callbacks_t __mpi_matrix_coord_full_callbacks = {
        __mpi_matrix_coord_full_element_count,
        __mpi_matrix_coord_full_index_status,
        __mpi_matrix_coord_full_index_reduce,
        __mpi_matrix_coord_full_index_to_offset
    };

//
////
//

base_int_t
__mpi_matrix_coord_upper_triangular_element_count(
    mpi_matrix_coord_ptr    coord
)
{
    if ( coord->dimensions.j <= coord->dimensions.i ) {
        //  X X X
        //  0 X X       a square triangle of dim j
        //  0 0 X
        //  0 0 0
        return (coord->dimensions.j * coord->dimensions.j + coord->dimensions.j) / 2;
    } else {
        //  X X X X
        //  0 X X X     a square triangle of dim i, with additional
        //  0 0 X X     columns of i elements
        return (coord->dimensions.i * coord->dimensions.i + coord->dimensions.i) / 2 + 
                (coord->dimensions.i * (coord->dimensions.j - coord->dimensions.i));
    }
}

mpi_matrix_coord_status_t
__mpi_matrix_coord_upper_triangular_index_status(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_coord_status_t   s = mpi_matrix_coord_status_is_defined | mpi_matrix_coord_status_is_unique;
    
    if ( p.i > p.j ) {
        // If transposed then it's a double-swap:
        if ( orient == mpi_matrix_orient_normal ) s = mpi_matrix_coord_status_is_defined;
    }
    return s;
}

bool
__mpi_matrix_coord_upper_triangular_index_reduce(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              *p
)
{
    base_int_t              t;
    
    // Swap indices if we're in the wrong half:
    if ( p->i > p->j ) {
        // Avoid a double swap:
        if ( orient == mpi_matrix_orient_normal ) t = p->i, p->i = p->j, p->j = t;
    } else if ( orient != mpi_matrix_orient_normal ) t = p->i, p->i = p->j, p->j = t;
    return true;
}

base_int_t
__mpi_matrix_coord_upper_triangular_index_to_offset(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{ 
    // Swap indices if we're in the wrong half:
    if ( p.i > p.j ) {
        // Avoid a double swap:
        if ( orient == mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    } else if ( orient != mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    if ( coord->is_row_major )
        return (p.j - p.i) + (p.i * (2 * coord->dimensions.i - p.i + 1)) / 2;
    return p.i + (p.j * p.j + p.j) / 2;
}

mpi_matrix_coord_callbacks_t __mpi_matrix_coord_upper_triangular_callbacks = {
        __mpi_matrix_coord_upper_triangular_element_count,
        __mpi_matrix_coord_upper_triangular_index_status,
        __mpi_matrix_coord_upper_triangular_index_reduce,
        __mpi_matrix_coord_upper_triangular_index_to_offset
    };

//
////
//

base_int_t
__mpi_matrix_coord_lower_triangular_element_count(
    mpi_matrix_coord_ptr    coord
)
{
    if ( coord->dimensions.j <= coord->dimensions.i ) {
        //  X 0 0
        //  X X 0       a square triangle of dim j, with additional
        //  X X X       rows of j elements
        //  X X X
        return (coord->dimensions.j * coord->dimensions.j + coord->dimensions.j) / 2 + 
                (coord->dimensions.j * (coord->dimensions.i - coord->dimensions.j));
    } else {
        //  X 0 0 0
        //  X X 0 0     a square triangle of dim i
        return (coord->dimensions.i * coord->dimensions.i + coord->dimensions.i) / 2;
    }
}

mpi_matrix_coord_status_t
__mpi_matrix_coord_lower_triangular_index_status(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_coord_status_t   s = mpi_matrix_coord_status_is_defined | mpi_matrix_coord_status_is_unique;
    
    if ( p.j > p.i ) {
        // If transposed then it's a double-swap:
        if ( orient == mpi_matrix_orient_normal ) s = mpi_matrix_coord_status_is_defined;
    }
    return s;
}

bool
__mpi_matrix_coord_lower_triangular_index_reduce(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              *p
)
{
    base_int_t              t;
    
    // Swap indices if we're in the wrong half:
    if ( p->j > p->i ) {
        // Avoid a double swap:
        if ( orient == mpi_matrix_orient_normal ) t = p->i, p->i = p->j, p->j = t;
    } else if ( orient != mpi_matrix_orient_normal ) t = p->i, p->i = p->j, p->j = t;
    return true;
}

base_int_t
__mpi_matrix_coord_lower_triangular_index_to_offset(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    // Swap indices if we're in the wrong half:
    if ( p.j > p.i ) {
        // Avoid a double swap:
        if ( orient == mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    } else if ( orient != mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    if ( coord->is_row_major )
        return p.j + (p.i * p.i + p.i) / 2;
    return (p.i - p.j) + (p.j * (2 * coord->dimensions.i - p.j + 1)) / 2;
}

mpi_matrix_coord_callbacks_t __mpi_matrix_coord_lower_triangular_callbacks = {
        __mpi_matrix_coord_lower_triangular_element_count,
        __mpi_matrix_coord_lower_triangular_index_status,
        __mpi_matrix_coord_lower_triangular_index_reduce,
        __mpi_matrix_coord_lower_triangular_index_to_offset
    };

//
////
//

base_int_t
__mpi_matrix_coord_band_diagonal_element_count(
    mpi_matrix_coord_ptr    coord
)
{
//
//  N_base = elements in square virtual matrix = rows * cols
//  N_1 = triangular matrix dim k1 = (k1^2+k1)/2
//  N_1 = triangular matrix dim k2 = (k2^2+k2)/2
//
//  N_total = N_base - N_1 - N_2
//
//          __________________
//     0  1| 2  3  4  5  -  - | -  -  -
//     -  6| 7  8  9 10 11  - | -  -  -
//     -  -|12 13 14 15 16 17 | -  -  -
//     -  -| - 18 19 20 21 22 |23  -  -
//     -  -| -  - 24 25 26 27 |28 29  -
//     -  -| -  -  - 30 31 32 |33 34 35
//         |__________________|
//
// d = (1 + 2 + 3) = 6
// N_base = 36
// N_1 = (2^2+2)/2 = 3
// N_2 = (3^2+3)/2 = 6
// N_total = 36 - 3 - 6 = 27
//
    base_int_t          d = (1 + coord->k1 + coord->k2);
    
    
    return (coord->is_row_major) ? 
                (d * coord->dimensions.i) - (coord->k1 * coord->k1 + coord->k1)/2 - (coord->k2 * coord->k2 + coord->k2)/2 :
                (d * coord->dimensions.j) - (coord->k1 * coord->k1 + coord->k1)/2 - (coord->k2 * coord->k2 + coord->k2)/2;
}

mpi_matrix_coord_status_t
__mpi_matrix_coord_band_diagonal_index_status(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_coord_status_t           s = mpi_matrix_coord_status_is_defined | mpi_matrix_coord_status_is_unique;
    
    if ( orient != mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    
    if ( coord->is_row_major ) {
        if ( (p.j < p.i - coord->k1) || (p.j > p.i + coord->k2) ) s = 0;
    } else {
        if ( (p.i < p.j - coord->k2) || (p.i > p.j + coord->k1) ) s = 0;
    }
    return s;
}

bool
__mpi_matrix_coord_band_diagonal_index_reduce(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              *p
)
{
    if ( orient != mpi_matrix_orient_normal ) {
        if ( (coord->is_row_major && (p->i < p->j - coord->k1) || (p->i > p->j + coord->k2)) ||
             (!coord->is_row_major && (p->j < p->i - coord->k2) || (p->j > p->i + coord->k1)) )
        {
            *p = int_pair_make(p->j, p->i);
            return true;
        }
    } else {
        if ( (coord->is_row_major && (p->j < p->i - coord->k1) || (p->j > p->i + coord->k2)) ||
             (!coord->is_row_major && (p->i < p->j - coord->k2) || (p->i > p->j + coord->k1)) ) return true;
    }
    return false;
}

base_int_t
__mpi_matrix_coord_band_diagonal_index_to_offset(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t orient,
    int_pair_t              p
)
{
//  d = (1 + k1 + k2)
//  i_hi = rows - k2
//  O_base = d * i + j + (k1 - i)
//  if ( i < k1 ) O_base -= (k1^2+k1)/2 - ((k1-i-1)^2+k1-i-1)/2
//  if ( i >= k1 ) O_base -= (k1^2+k1)/2
//  if ( i > i_hi ) O_base -= ((i-i_hi)^2+i-i_hi)/2
//          __________________
//     0  1| 2  3  4  5  -  - | -  -  -
//     -  6| 7  8  9 10 11  - | -  -  -
//     -  -|12 13 14 15 16 17 | -  -  -
//     -  -| - 18 19 20 21 22 |23  -  -
//     -  -| -  - 24 25 26 27 |28 29  -
//     -  -| -  -  - 30 31 32 |33 34 35
//         |__________________|
//
// d = (1 + 2 + 3) = 6
// i_hi = 6 - 3 = 3
//
// (0, 0):
// O_base = 6 * 0 + 0 + (2 - 0) = 2
// if ( 0 < 2 ) O_base -= (4+2)/2 - ((1^2 + 2 -1)/2) = 2 - (3 - 2/2) = 2 - 2 = 0
//
// (0, 1):
// O_base = 6 * 0 + 1 + (2 - 0) = 3
// if ( 0 < 2 ) O_base -= (4+2)/2 - ((1^2 + 2 -1)/2) = 3 - (3 - 2/2) = 3 - 2 = 1
//
// (1, 0):
// O_base = 6 * 1 + 0 + (2 - 1) = 7
// if ( 1 < 2 ) O_base -= (4+2)/2 - ((2-1-1)^2+2-1-1)/2 = 7 - (3 - 0) = 7 - 3 = 4
//
// (1, 3):
// O_base = 6 * 1 + 3 + (2 - 1) = 10
// if ( 1 < 2 ) O_base -= (4+2)/2 - ((2-1-1)^2+2-1-1)/2 = 10 - (3 - 0) = 10 - 3 = 7
//
// (2,0):
// O_base = 6 * 2 + 0 + (2 - 2) = 12
// if ( 2 >= 2 ) O_base -= (2^2+2)/2 = 12 - 3 = 9
//
// (3,1):
// O_base = 6 * 3 + 1 + (2 - 3) = 18 + 1 - 1 = 18
// if ( 3 >= 2 ) O_base -= (2^2+2)/2 = 18 - 3 = 15
//
// (4, 2):
// O_base = 6 * 4 + 2 + (2 - 4) = 24 + 2 - 2 = 24
// if ( 4 >= 2 ) O_base -= (2^2+2)/2 = 24 - 3 = 21
// if ( 4 > 3 ) O_base -= ((4-3)^2+4-3)/2 = 21 - (1+1)/2 = 21 - 1 = 20
//
// (4, 4):
// O_base = 6 * 4 + 4 + (2 - 4) = 24 + 2 - 2 = 26
// if ( 4 >= 2 ) O_base -= (2^2+2)/2 = 26 - 3 = 23
// if ( 4 > 3 ) O_base -= ((4-3)^2+4-3)/2 = 23 - (1+1)/2 = 23 - 1 = 22
//
// (5,5):
// O_base = 6 * 5 + 5 + (2 - 5) = 30 + 5 - 3 = 32
// if ( 5 >= 2 ) O_base -= (2^2+2)/2 = 32 - 3 = 29
// if ( 5 > 3 ) O_base -= ((5-3)^2+5-3)/2 = 29 - (4+2)/2 = 29 - 3 = 26
//
    base_int_t      d = (1 + coord->k1 + coord->k2);
    base_int_t      offset = -1, tmp;
    
    if ( orient != mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    if ( coord->is_row_major ) {
        if ( (p.j >= p.i - coord->k1) && (p.j <= p.i + coord->k2) ) {
            base_int_t      i_hi = coord->dimensions.i - coord->k2;
            
            offset = (d - 1) * p.i + p.j + coord->k1;
            if ( p.i < coord->k1 - 1 ) {
                tmp = coord->k1 - p.i - 1;
                offset -= (coord->k1 * coord->k1 + coord->k1)/2 - (tmp * tmp + tmp)/2;
            } else {
                offset -= (coord->k1 * coord->k1 + coord->k1)/2;
            }
            if ( p.i > i_hi ) {
                tmp = p.i - i_hi;
                offset -= (tmp * tmp + tmp)/2;
            }
        }
    } else if ( (p.i >= p.j - coord->k2) && (p.i <= p.j + coord->k1) ) {
        base_int_t      j_hi = coord->dimensions.j - coord->k1;
        
        offset = (d - 1) * p.j + p.i + coord->k2;
        if ( p.j < coord->k2 - 1 ) {
            tmp = coord->k2 - p.j - 1;
            offset -= (coord->k2 * coord->k2 + coord->k2)/2 - (tmp * tmp + tmp)/2;
        } else {
            offset -= (coord->k2 * coord->k2 + coord->k2)/2;
        }
        if ( p.j > j_hi ) {
            tmp = p.j - j_hi;
            offset -= (tmp * tmp + tmp)/2;
        }
    }
    return offset;
}

mpi_matrix_coord_callbacks_t __mpi_matrix_coord_band_diagonal_callbacks = {
        __mpi_matrix_coord_band_diagonal_element_count,
        __mpi_matrix_coord_band_diagonal_index_status,
        __mpi_matrix_coord_band_diagonal_index_reduce,
        __mpi_matrix_coord_band_diagonal_index_to_offset
    };

//
////
//

base_int_t
__mpi_matrix_coord_tridiagonal_element_count(
    mpi_matrix_coord_ptr    coord
)
{
    if ( coord->dimensions.j == coord->dimensions.i ) {
        //  X X 0 0
        //  X X X 0
        //  0 X X X
        //  0 0 X X
        //return 4 + 3 ( coord->dimensions.j - 2);
        return 3 * coord->dimensions.j - 2;
    }
    else if ( coord->dimensions.j <= coord->dimensions.i ) {
        //  X X 0
        //  X X X
        //  0 X X
        //  0 0 X
        return (3 * coord->dimensions.j - 2) + 1;
    }
    else {
        //  X X 0 0
        //  X X X 0
        //  0 X X X
        return (3 * coord->dimensions.i - 2) + 1;
    }
}

mpi_matrix_coord_status_t
__mpi_matrix_coord_tridiagonal_index_status(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_coord_status_t   s = mpi_matrix_coord_status_is_defined | mpi_matrix_coord_status_is_unique;
    int_pair_t                  P = (orient != mpi_matrix_orient_normal) ? int_pair_make(p.j, p.i) : p;
    base_int_t                  d_ij = P.i - P.j;;
    
    // Valid elements of a triadiagonal matrix are such that |i-j| <= 1
    if ( ((d_ij >= 0) ? d_ij : -d_ij) > 1 ) s = 0;
    return s;
}

bool
__mpi_matrix_coord_tridiagonal_index_reduce(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              *p
)
{
    int_pair_t              P = (orient != mpi_matrix_orient_normal) ? int_pair_make(p->j, p->i) : *p;
    base_int_t              d_ij = P.i - P.j;;
    
    // Valid elements of a triadiagonal matrix are such that |i-j| <= 1
    if ( ((d_ij >= 0) ? d_ij : -d_ij) <= 1 ) {
        *p = P;
        return true;
    }
    return false;
}

base_int_t
__mpi_matrix_coord_tridiagonal_index_to_offset(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    base_int_t              d_ij;
    
    if ( orient != mpi_matrix_orient_normal ) p = int_pair_make(p.j, p.i);
    
    // Valid elements of a triadiagonal matrix are such that |i-j| <= 1
    d_ij = p.i - p.j;
    if ( ((d_ij >= 0) ? d_ij : -d_ij) <= 1 ) {
        if ( coord->is_row_major ) {
            return p.i * 3 + (p.j - p.i);
        } else {
            return p.j * 3 + (p.i - p.j);
        }
    }
    return -1;
}

mpi_matrix_coord_callbacks_t __mpi_matrix_coord_tridiagonal_callbacks = {
        __mpi_matrix_coord_tridiagonal_element_count,
        __mpi_matrix_coord_tridiagonal_index_status,
        __mpi_matrix_coord_tridiagonal_index_reduce,
        __mpi_matrix_coord_tridiagonal_index_to_offset
    };

//
////
//

base_int_t
__mpi_matrix_coord_diagonal_element_count(
    mpi_matrix_coord_ptr    coord
)
{
    return (coord->dimensions.j <= coord->dimensions.i) ? coord->dimensions.j : coord->dimensions.i;
}

mpi_matrix_coord_status_t
__mpi_matrix_coord_diagonal_index_status(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t orient,
    int_pair_t              p
)
{
    return (p.i == p.j) ? (mpi_matrix_coord_status_is_defined | mpi_matrix_coord_status_is_unique) : 0;
}

bool
__mpi_matrix_coord_diagonal_index_reduce(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t orient,
    int_pair_t              *p
)
{
    return ( p->i == p->j );
}

base_int_t
__mpi_matrix_coord_diagonal_index_to_offset(
    mpi_matrix_coord_ptr    coord,
    mpi_matrix_orient_t orient,
    int_pair_t              p
)
{
    if ( p.i == p.j ) return p.i;
    return -1;
}

mpi_matrix_coord_callbacks_t __mpi_matrix_coord_diagonal_callbacks = {
        __mpi_matrix_coord_diagonal_element_count,
        __mpi_matrix_coord_diagonal_index_status,
        __mpi_matrix_coord_diagonal_index_reduce,
        __mpi_matrix_coord_diagonal_index_to_offset
    };

//

mpi_matrix_coord_ptr
__mpi_matrix_coord_init(
    mpi_matrix_coord_ptr        coord,
    mpi_matrix_coord_type_t     type,
    bool                        is_row_major,
    int_pair_t                  dimensions,
    va_list                     argv
)
{
    mpi_matrix_coord_t          *COORD = (mpi_matrix_coord_t*)coord;
    
    COORD->type = type;
    COORD->is_row_major = is_row_major;
    COORD->dimensions = dimensions;
    switch ( type ) {
        case mpi_matrix_coord_type_full:
            COORD->callbacks = __mpi_matrix_coord_full_callbacks;
            break;
        case mpi_matrix_coord_type_upper_triangular:
            COORD->callbacks = __mpi_matrix_coord_upper_triangular_callbacks;
            break;
        case mpi_matrix_coord_type_lower_triangular:
            COORD->callbacks = __mpi_matrix_coord_lower_triangular_callbacks;
            break;
        case mpi_matrix_coord_type_band_diagonal: {
            if ( dimensions.i == dimensions.j ) {
                base_int_t      k1 = va_arg(argv, base_int_t),
                                k2 = va_arg(argv, base_int_t);
                                
                if ( (k1 >= 0) && (k2 >= 0) && (k1 + k2 <= dimensions.i) ) {
                    COORD->k1 = k1;
                    COORD->k2 = k2;
                    COORD->callbacks = __mpi_matrix_coord_band_diagonal_callbacks;
                } else {
                    fprintf(stderr, 
                        "%s:%d : invalid band width for band-diagonal matrix (max width=" BASE_INT_FMT " for k1=" BASE_INT_FMT ", k2=" BASE_INT_FMT ")\n",
                        __FILE__, __LINE__, dimensions.i, k1, k2);
                    coord = NULL;
                }
            } else {
                fprintf(stderr, "%s:%d : band-diagonal matrices must be square, dimensions were (" BASE_INT_FMT "," BASE_INT_FMT ")\n",
                            __FILE__, __LINE__, dimensions.i, dimensions.j);
                coord = NULL;
            }
            break;
        }
        case mpi_matrix_coord_type_tridiagonal:
            COORD->callbacks = __mpi_matrix_coord_tridiagonal_callbacks;
            break;
        case mpi_matrix_coord_type_diagonal:
            COORD->callbacks = __mpi_matrix_coord_diagonal_callbacks;
            break;
        case mpi_matrix_coord_type_max:
            break;
    }
    return coord;
}

//

mpi_matrix_coord_ptr
mpi_matrix_coord_create(
    mpi_matrix_coord_type_t     type,
    bool                        is_row_major,
    int_pair_t                  dimensions,
    ...
)
{
    mpi_matrix_coord_ptr         out_coord = NULL;
    
    if ( type >= 0 && type < mpi_matrix_coord_type_max ) {
        mpi_matrix_coord_t      *new_coord = (mpi_matrix_coord_t*)malloc(sizeof(mpi_matrix_coord_t));
        
        if ( new_coord ) {
            va_list             argv;
            
            va_start(argv, dimensions);
            out_coord = (mpi_matrix_coord_ptr)__mpi_matrix_coord_init(
                                                    (mpi_matrix_coord_ptr)new_coord,
                                                    type,
                                                    is_row_major,
                                                    dimensions,
                                                    argv);
            va_end(argv);
            if ( ! out_coord ) free((void*)new_coord);
        }
    }
    return out_coord;
}

//

mpi_matrix_coord_ptr
mpi_matrix_coord_init(
    mpi_matrix_coord_ptr        coord,
    mpi_matrix_coord_type_t     type,
    bool                        is_row_major,
    int_pair_t                  dimensions,
    ...
)
{
    mpi_matrix_coord_ptr        out_coord;
    va_list                     argv;
    
    va_start(argv, dimensions);
    out_coord = (mpi_matrix_coord_ptr)__mpi_matrix_coord_init(
                                            coord,
                                            type,
                                            is_row_major,
                                            dimensions,
                                            argv);
    va_end(argv);
    return out_coord;
}

//

void
mpi_matrix_coord_destroy(
    mpi_matrix_coord_ptr    coord
)
{
    free((void*)coord);
}
