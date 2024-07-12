/*
 *
 */
 
#include "mpi_matrix_dist_coord.h"
#include "mpi_utils.h"

//

static inline int
__mpi_matrix_dist_coord_grid_part_dist(
    int     i,
    int     j
)
{
    if ( i > j ) return (i - j);
    return (j - i);
}

//

mpi_matrix_dist_coord_ptr
mpi_matrix_dist_coord_create(
    mpi_matrix_dist_coord_split_t   kind_of_split,
    mpi_matrix_coord_t              *global_coord,
    MPI_Comm                        global_comm,
    mpi_matrix_dist_coord_hint_t    kind_of_hint,
    int_pair_t                      *dist_hint,
    const char*                     *error_msg
)
{
    int                             global_size, global_rank;
    int                             local_size, local_rank, dist_size;
    MPI_Comm                        local_comm;
    int_pair_t                      block_dims;
    bool                            local_comm_is_set = false;
    
    MPI_Comm_size(global_comm, &global_size);
    MPI_Comm_rank(global_comm, &global_rank);
    if ( kind_of_split == mpi_matrix_dist_coord_split_by_rank ) {
        /*
         * If there's no split, then the local size/rank is just one
         * and the distribution rank count is the global size; there
         * is no need for a local_comm since each dist rank will NOT
         * be sharing responsiblity with other ranks:
         */
        dist_size = global_size;
        local_rank = 0;
        local_size = 1;
    } else {
        int         split_type;
        
        switch ( kind_of_split ) {
#ifdef OPEN_MPI
            case mpi_matrix_dist_coord_split_by_node:
                split_type = OMPI_COMM_TYPE_NODE;
                break;
            case mpi_matrix_dist_coord_split_by_socket:
                split_type = OMPI_COMM_TYPE_SOCKET;
                break;
            case mpi_matrix_dist_coord_split_by_numa:
                split_type = OMPI_COMM_TYPE_NUMA;
                break;
#else
            case mpi_matrix_dist_coord_split_by_node:
                split_type = MPI_COMM_TYPE_SHARED;
                break;
#endif
            case mpi_matrix_dist_coord_split_by_rank:
            case mpi_matrix_dist_coord_split_max:
                if ( error_msg ) *error_msg = "invalid kind of split";
                return NULL;
        }
        MPI_Comm_split_type(global_comm, split_type, global_rank, MPI_INFO_NULL, &local_comm);
        local_comm_is_set = true;
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_rank(local_comm, &local_rank);
        
        /* Figure out the number of domains in the split -- equal to the number
         * of ranks that are root in their local_comm:
         */
        dist_size = (local_rank == 0);
        if ( global_rank == 0 )
            MPI_Reduce(MPI_IN_PLACE, &dist_size, 1, MPI_INT, MPI_SUM, 0, global_comm);
        else
            MPI_Reduce(&dist_size, NULL, 1, MPI_INT, MPI_SUM, 0, global_comm);
        MPI_Bcast(&dist_size, 1, MPI_INT, 0, global_comm);
    }

    /*
     * At this point we know that the matrix must be split across dist_size
     * ranks in the MPI runtime.  Determine how the global matrix will be
     * decomposed into sub-matrix blocks based on the hint et al.
     *
     * Note that this is easiest for a full coordinate system (no symmetry)
     * but gets more complicated for other types.
     */
    switch ( global_coord->type ) {
    
        case mpi_matrix_coord_type_full: {
            mpi_matrix_dist_coord_t     *new_coord = NULL;
            size_t                      dist_root_ranks_bytes = ((dist_size == global_size) ? 0 : dist_size) * sizeof(int);
            base_int_t                  block_i, block_j;
            base_int_t                  block_rows_full, block_cols_full;
            base_int_t                  block_rows_part, block_cols_part;
            
            switch ( kind_of_hint ) {
                case mpi_matrix_dist_coord_hint_none: {
                    double      whole_part, fract_part;

                    fract_part = modf(sqrt((double)dist_size), &whole_part);
                    if ( fabs(fract_part) < FLT_EPSILON ) {
                        /*
                         * Perfect square:
                         */
                        block_dims = int_pair_make((int)whole_part, (int)whole_part);
                    } else {
                        static int  grid_factors[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 0};
                        int         i, j;
                    
                        /*
                         * Auto-grid:
                         */
                        i = dist_size, j = 1;
                        while ( i > j ) {
                            int     *factors = grid_factors;
                        
                            while ( *factors ) {
                                if ( (i % *factors) == 0 ) {
                                    // If the two values get further apart, skip it and exit now:
                                    int     li = i / *factors, lj = j * *factors;

                                    if ( __mpi_matrix_dist_coord_grid_part_dist(li, lj) > __mpi_matrix_dist_coord_grid_part_dist(i, j) ) {
                                        li = i; i = j; j = li;
                                    } else {
                                        i = li; j = lj;
                                    }
                                    break;
                                }
                                factors++;
                            }
                            if ( *factors == 0 ) break;
                        }
                        block_dims = int_pair_make(i, j);
                    }
                    break;
                }
            
                case mpi_matrix_dist_coord_hint_exact: {
                    /*
                     * The product of the two dimensions in dist_hint must equal
                     * the dist_size:
                     */
                    if ( !dist_hint || (dist_hint->i * dist_hint->j != dist_size) ) {
                        if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                        if ( error_msg ) *error_msg = "exact split dims do not match dist size";
                        return NULL;
                    }
                    block_dims = *dist_hint;
                    break;
                }
        
                case mpi_matrix_dist_coord_hint_ratio: {
                    /*
                     * The dist_hint provides the ratio of block rows to
                     * columns that should be multipied to reach dist_size;
                     * if either dimension is zero it is implied that a fixed
                     * block count of 1 in that dimension is desired:
                     */
                    if ( ! dist_hint ) {
                        if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                        if ( error_msg ) *error_msg = "no ratio hint supplied";
                        return NULL;
                    }
                    if ( dist_hint->i == 0 ) {
                        if ( dist_hint->j == 0 ) {
                            /* Single block, period; works only if dist_size is also one: */
                            if ( dist_size != 1 ) {
                                if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                                if ( error_msg ) *error_msg = "single block hinted, dist size != 1";
                                return NULL;
                            }
                            block_dims = int_pair_make(1, 1);   
                        } else {
                            block_dims = int_pair_make(1, dist_size);
                        }
                    }
                    else if ( dist_hint->j == 0 ) {
                        block_dims = int_pair_make(dist_size, 1);
                    }
                    else {
                        /*
                         * dist_size = (dist_hint->i/dist_hint->j * j) * j
                         * dist_size = j^2 * (dist_hint->i/dist_hint->j)
                         * j = Sqrt(dist_size * dist_hint->j / dist_hint->i)
                         * i = dist_size/j
                         *
                         * dist_size = 8
                         * dist_hint = { .i = 3, .j = 2 };
                         * j = floor(Sqrt(8 * ( 2 / 3))) = floor(Sqrt(16/3)) = floor(2.31) = 2
                         * i = 8 / 2 = 4
                         *    -OR-
                         * j = ceil(Sqrt(8 * (2 / 3))) = ceil(2.31) = 3
                         * i = 8 / 3 = 2
                         *
                         * dist_size = 10
                         * dist_hint = { .i = 4, .j = 1 };
                         * j = floor(Sqrt(10 * 1 / 4)) = floor(Sqrt(2.5)) = floor(1.58) = 1
                         * i = 10 / 1 = 10
                         *   -OR-
                         * j = ceil(Sqrt(10 * 1 / 4)) = ceil(1.58) = 2
                         * i = 10 / 2 = 5
                         */
                        float       ratio = (float)dist_hint->j / (float)dist_hint->i;
                        float       j_exact = sqrtf(((float)dist_size * ratio));
                        int         j_lo = floorf(j_exact), j_hi = ceilf(j_exact);
                        int         i_lo = floorf((float)dist_size / j_hi),
                                    i_hi = floorf((float)dist_size / j_lo);
                        
                        if ( i_lo * j_hi != dist_size ) {
                            if ( i_hi * j_lo != dist_size ) {
                                /* Neither pair works w.r.t. the dist_size: */
                                if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                                if ( error_msg ) *error_msg = "no 2d block dim matching dist size";
                                return NULL;
                            }
                            block_dims = int_pair_make(i_hi, j_lo);
                        }
                        else if ( i_hi * j_lo != dist_size ) {
                            block_dims = int_pair_make(i_lo, j_hi);
                        }
                        else {
                            /* Both pairs yield the appropriate rank count.  Which is
                             * compatible with the hint?
                             */
                            if ( dist_hint->i > dist_hint->j ) {
                                if ( i_hi > j_lo )
                                    block_dims = int_pair_make(i_hi, j_lo);
                                else if ( i_lo > j_hi )
                                    block_dims = int_pair_make(i_lo, j_hi);
                                else if ( __mpi_matrix_dist_coord_grid_part_dist(i_hi, j_lo) <= __mpi_matrix_dist_coord_grid_part_dist(i_lo, j_hi) )
                                    block_dims = int_pair_make(i_hi, j_lo);
                                else
                                    block_dims = int_pair_make(i_lo, j_hi);
                            }
                            else if ( dist_hint->i < dist_hint->j ) {
                                if ( i_hi < j_lo )
                                    block_dims = int_pair_make(i_hi, j_lo);
                                else if ( i_lo < j_hi )
                                    block_dims = int_pair_make(i_lo, j_hi);
                                else if ( __mpi_matrix_dist_coord_grid_part_dist(i_hi, j_lo) <= __mpi_matrix_dist_coord_grid_part_dist(i_lo, j_hi) )
                                    block_dims = int_pair_make(i_hi, j_lo);
                                else
                                    block_dims = int_pair_make(i_lo, j_hi);
                            } else if ( __mpi_matrix_dist_coord_grid_part_dist(i_hi, j_lo) <= __mpi_matrix_dist_coord_grid_part_dist(i_lo, j_hi) ) {
                                block_dims = int_pair_make(i_hi, j_lo);
                            }
                            else {
                                block_dims = int_pair_make(i_lo, j_hi);
                            }
                        }
                    }
                    break;
                }
                case mpi_matrix_dist_coord_hint_max:
                    if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                    if ( error_msg ) *error_msg = "invalid kind of hint";
                    return NULL;
                
            }
                
            /*
             * At this point block_dims describes the block-cyclic distribution
             * to be adopted:
             */
            new_coord = (mpi_matrix_dist_coord_t*)malloc(sizeof(mpi_matrix_dist_coord_t) + dist_root_ranks_bytes);
            if ( ! new_coord ) {
                if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                if ( error_msg ) *error_msg = "unable to allocate mpi_matrix_dist_coord";
                return NULL;
            }
            
            /*
             * Set global properties:
             */
            new_coord->global_coord = *global_coord;
            new_coord->global_comm = global_comm;
            new_coord->global_rank = global_rank;
            new_coord->global_size = global_size;
            
            /*
             * Set dist roots properties:
             */
            new_coord->dist_root_size = dist_size;
            new_coord->dist_root_idx = -1;
            if ( dist_root_ranks_bytes > 0 ) {
                MPI_Comm_split(global_comm, (local_rank == 0) ? 0 : MPI_UNDEFINED, global_rank, &new_coord->dist_root_comm);
                if ( local_rank == 0 ) MPI_Comm_rank(new_coord->dist_root_comm, &new_coord->dist_root_idx);
                
                /*
                 * Each dist root broadcasts its dist root index to its peers:
                 */
                MPI_Bcast(&new_coord->dist_root_idx, 1, MPI_INT, 0, local_comm);
                
                /*
                 * Each rank allocates storage for the dist roots rank list:
                 */
                new_coord->dist_root_ranks = (int*)((void*)new_coord + sizeof(mpi_matrix_dist_coord_t));
                
                /*
                 * Have each rank in the dist_root_comm add its global_rank to the
                 * dist_root_ranks list (in sequence) and collate among the dist root ranks:
                 */
                if ( local_rank == 0 )
                    MPI_Allgather(&global_rank, 1, MPI_INT, new_coord->dist_root_ranks, 1, MPI_INT, new_coord->dist_root_comm);
                
                /*
                 * Now have the root of each local comm broadcast this list to its peers so that every rank
                 * knows which global rank corresponds to which distributed block:
                 */
                MPI_Bcast(new_coord->dist_root_ranks, dist_size, MPI_INT, 0, local_comm);
            } else {
                /*
                 * If the distribution is done by global rank and not some
                 * reduced number of ranks then the sequence [0,global_size) is the
                 * implied set of distributed root ranks.  No need to allocate
                 * memory for that, we'll use callbacks that ignore the dist_root_ranks
                 * anyway:
                 */
                new_coord->dist_root_ranks = NULL;
                new_coord->dist_root_idx = global_rank;
            }
            
            /*
             * Set local properties:
             */
            new_coord->local_comm = local_comm_is_set ? local_comm : global_comm;
            new_coord->local_rank = local_rank;
            new_coord->local_size = local_size;
            
            /*
             * Is the global coordinate system row- or column-major?  That will determine
             * the assignment of indices to ranks:
             */
            block_rows_full = global_coord->dimensions.i / block_dims.i;
            block_rows_part = global_coord->dimensions.i % block_dims.i;
            block_cols_full = global_coord->dimensions.j / block_dims.j;
            block_cols_part = global_coord->dimensions.j % block_dims.j;
            if ( global_coord->is_row_major ) {
                block_i = new_coord->dist_root_idx / block_dims.j;
                block_j = new_coord->dist_root_idx % block_dims.j;
            } else {
                block_j = new_coord->dist_root_idx / block_dims.i;
                block_i = new_coord->dist_root_idx % block_dims.i;
            }
            mpi_matrix_coord_init(&new_coord->local_coord,
                    mpi_matrix_coord_type_full,
                    global_coord->is_row_major,
                    int_pair_make(block_rows_full + ((block_i < block_rows_part) ? 1 : 0),
                                  block_cols_full + ((block_j < block_cols_part) ? 1 : 0))
                );
            if ( block_i < block_rows_part ) block_rows_part = block_i;
            if ( block_j < block_cols_part ) block_cols_part = block_j;
            new_coord->local_index = int_pair_make(block_rows_full * block_i + block_rows_part,
                                                   block_cols_full * block_j + block_cols_part);
            
            return (mpi_matrix_dist_coord_ptr)new_coord;
        }
    
        /*
         * The triangular forms restrict the block distribution to be a mix of
         * triangular blocks on the diagonal and rectangular blocks off-diagonal.
         *
         * At the most basic, the dist_size must be equated with a number of blocks
         * counted as N_block = n_block(n_block+1)/2, where n_block is the count in
         * both row and column directions and R is the number of ranks:
         *
         *     R = n_block(n_block+1)/2
         *     2R = n_block(n_block+1)
         *     n_block^2 + n_block - 2R = 0
         *     n_block = (-1 ± Sqrt[1 + 8R])/2
         *
         * The only solution that makes sense (is positive) is
         *
         *     n_block = (-1 + Sqrt[1 + 8R])/2
         *
         * For example:
         *
         *     R = 6 ranks
         *     N_block = (-1 + Sqrt[1+48])/2
         *             = (-1 + 7)/2 = 3
         *
         *         []0000
         *         [][]00
         *         [][][]
         * 
         */
        case mpi_matrix_coord_type_upper_triangular:
        case mpi_matrix_coord_type_lower_triangular: {
            switch ( kind_of_hint ) {
                case mpi_matrix_dist_coord_hint_none: {
                    float       sqrt_term, N_block_whole, N_block_fract;
                    
                    sqrt_term = sqrtf(1.0f + 8.0f * dist_size);
                    N_block_fract = modff(0.5f * (sqrt_term - 1.0f), &N_block_whole);
                    if ( fabs(N_block_fract) > FLT_EPSILON ) {
                        if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                        if ( error_msg ) *error_msg = "no triangular block dim matching dist size";
                        return NULL;
                    }
                    block_dims = int_pair_make(N_block_whole, N_block_whole);
                    break;
                }
            
                case mpi_matrix_dist_coord_hint_exact:
                case mpi_matrix_dist_coord_hint_ratio:
                case mpi_matrix_dist_coord_hint_max:
                    if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
                    if ( error_msg ) *error_msg = "invalid kind of hint for triangular coordinates";
                    return NULL;
            }
            break;
        }
        
        default:
            if ( local_comm_is_set ) MPI_Comm_free(&local_comm);
            if ( error_msg ) *error_msg = "unsupported coordinate system for distribution";
            return NULL;
    }
    
    return NULL;
}

//

void
mpi_matrix_dist_coord_destroy(
    mpi_matrix_dist_coord_ptr       dist_coord
)
{
    int                             comm_cmp;
    
    /*
     * Destroy any additional communicators:
     */
    MPI_Comm_compare(dist_coord->global_comm, dist_coord->local_comm, &comm_cmp);
    if ( comm_cmp != MPI_IDENT ) MPI_Comm_free(&dist_coord->local_comm);
    if ( (dist_coord->dist_root_size != dist_coord->global_size) && (dist_coord->local_rank == 0) ) MPI_Comm_free(&dist_coord->dist_root_comm);
}
