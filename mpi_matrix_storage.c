
#include "mpi_matrix_storage.h"

/*
 * For shared memory handling
 */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h> 

enum {
    mpi_matrix_storage_options_no_coord = 1 << 0,
    mpi_matrix_storage_options_is_immutable = 1 << 1,
    mpi_matrix_storage_options_is_mmap = 1 << 2
};

//

const size_t mpi_matrix_storage_datatype_byte_sizes[] = {
                    sizeof(float),
                    sizeof(double),
                    sizeof(float complex),
                    sizeof(double complex),
                    0
                };

//

const char*
mpi_matrix_storage_type_get_name(
    mpi_matrix_storage_type_t   type
)
{
    static const char   *type_names[] = {
                            "basic",
                            "sparse-bst",
                            "sparse-compressed",
                            "sparse-coordinate",
                            NULL
                        };
    if ( type >= 0 && type < mpi_matrix_storage_type_max ) return type_names[type];
    return NULL;
}

//

const char*
mpi_matrix_storage_datatype_get_name(
    mpi_matrix_storage_datatype_t   type
)
{
    static const char   *type_names[] = {
                            "single-prec real",
                            "double-prec real",
                            "single-prec complex",
                            "double-prec complex",
                            NULL
                        };
    if ( type >= 0 && type < mpi_matrix_storage_datatype_max ) return type_names[type];
    return NULL;
}

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    float                   *elements;
} mpi_matrix_storage_basic_real_sp_t;

size_t
__mpi_matrix_storage_basic_real_sp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    return sizeof(mpi_matrix_storage_basic_real_sp_t) + sizeof(float) * mpi_matrix_coord_element_count(&storage->coord);
}

bool
__mpi_matrix_storage_basic_real_sp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_basic_real_sp_t  *STORAGE = (mpi_matrix_storage_basic_real_sp_t*)storage;
    float                               *ELEMENT = (float*)element;
    base_int_t                          offset = mpi_matrix_coord_index_to_offset(
                                                        &storage->coord,
                                                        orient,
                                                        p);
    *ELEMENT = ( offset >= 0 ) ? STORAGE->elements[offset] : 0.0f;
    return true;
}

bool
__mpi_matrix_storage_basic_real_sp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) ) {
        mpi_matrix_storage_basic_real_sp_t  *STORAGE = (mpi_matrix_storage_basic_real_sp_t*)storage;
        float                               *ELEMENT = (float*)element;
        base_int_t                          offset = mpi_matrix_coord_index_to_offset(
                                                            &storage->coord,
                                                            orient,
                                                            p);
        if ( offset >= 0 ) {
            STORAGE->elements[offset] = *ELEMENT;
            return true;
        }
    }
    return false;
}

bool
__mpi_matrix_storage_basic_real_sp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    float                   zero = 0.0f;
    return __mpi_matrix_storage_basic_real_sp_set(storage, orient, p, &zero);
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_basic_real_sp_callbacks = {
        NULL,
        __mpi_matrix_storage_basic_real_sp_byte_usage,
        __mpi_matrix_storage_basic_real_sp_clear,
        __mpi_matrix_storage_basic_real_sp_get,
        __mpi_matrix_storage_basic_real_sp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    double                  *elements;
} mpi_matrix_storage_basic_real_dp_t;

size_t
__mpi_matrix_storage_basic_real_dp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    return sizeof(mpi_matrix_storage_basic_real_dp_t) + sizeof(double) * mpi_matrix_coord_element_count(&storage->coord);
}

bool
__mpi_matrix_storage_basic_real_dp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_basic_real_dp_t  *STORAGE = (mpi_matrix_storage_basic_real_dp_t*)storage;
    double                              *ELEMENT = (double*)element;
    base_int_t                          offset = mpi_matrix_coord_index_to_offset(
                                                        &storage->coord,
                                                        orient,
                                                        p);
    *ELEMENT = ( offset >= 0 ) ? STORAGE->elements[offset] : 0.0;
    return true;
}

bool
__mpi_matrix_storage_basic_real_dp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) ) {
        mpi_matrix_storage_basic_real_dp_t  *STORAGE = (mpi_matrix_storage_basic_real_dp_t*)storage;
        double                              *ELEMENT = (double*)element;
        base_int_t                          offset = mpi_matrix_coord_index_to_offset(
                                                            &storage->coord,
                                                            orient,
                                                            p);
        if ( offset >= 0 ) {
            STORAGE->elements[offset] = *ELEMENT;
            return true;
        }
    }
    return false;
}

bool
__mpi_matrix_storage_basic_real_dp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    double                  zero = 0.0;
    return __mpi_matrix_storage_basic_real_dp_set(storage, orient, p, &zero);
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_basic_real_dp_callbacks = {
        NULL,
        __mpi_matrix_storage_basic_real_dp_byte_usage,
        __mpi_matrix_storage_basic_real_dp_clear,
        __mpi_matrix_storage_basic_real_dp_get,
        __mpi_matrix_storage_basic_real_dp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    float complex           *elements;
} mpi_matrix_storage_basic_complex_sp_t;

size_t
__mpi_matrix_storage_basic_complex_sp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    return sizeof(mpi_matrix_storage_basic_complex_sp_t) + sizeof(float complex) * mpi_matrix_coord_element_count(&storage->coord);
}

bool
__mpi_matrix_storage_basic_complex_sp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_basic_complex_sp_t   *STORAGE = (mpi_matrix_storage_basic_complex_sp_t*)storage;
    float complex                           *ELEMENT = (float complex*)element;
    base_int_t                              offset = mpi_matrix_coord_index_to_offset(
                                                            &storage->coord,
                                                            orient,
                                                            p);
    *ELEMENT = ( offset >= 0 ) ? 
                    ( (orient == mpi_matrix_orient_conj_transpose) ? conjf(STORAGE->elements[offset]) : STORAGE->elements[offset] ) :
                    CMPLXF(0.0f, 0.0f);
    return true;
}

bool
__mpi_matrix_storage_basic_complex_sp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) ) {
        mpi_matrix_storage_basic_complex_sp_t   *STORAGE = (mpi_matrix_storage_basic_complex_sp_t*)storage;
        float complex                           *ELEMENT = (float complex*)element;
        base_int_t                              offset = mpi_matrix_coord_index_to_offset(
                                                                &storage->coord,
                                                                orient,
                                                                p);
        if ( offset >= 0 ) {
            STORAGE->elements[offset] = *ELEMENT;
            return true;
        }
    }
    return false;
}

bool
__mpi_matrix_storage_basic_complex_sp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    float complex           zero = CMPLXF(0.0f, 0.0f);
    return __mpi_matrix_storage_basic_complex_sp_set(storage, orient, p, &zero);
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_basic_complex_sp_callbacks = {
        NULL,
        __mpi_matrix_storage_basic_complex_sp_byte_usage,
        __mpi_matrix_storage_basic_complex_sp_clear,
        __mpi_matrix_storage_basic_complex_sp_get,
        __mpi_matrix_storage_basic_complex_sp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    double complex          *elements;
} mpi_matrix_storage_basic_complex_dp_t;

size_t
__mpi_matrix_storage_basic_complex_dp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    return sizeof(mpi_matrix_storage_basic_complex_dp_t) + sizeof(double complex) * mpi_matrix_coord_element_count(&storage->coord);
}

bool
__mpi_matrix_storage_basic_complex_dp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_basic_complex_dp_t   *STORAGE = (mpi_matrix_storage_basic_complex_dp_t*)storage;
    double complex                          *ELEMENT = (double complex*)element;
    base_int_t                              offset = mpi_matrix_coord_index_to_offset(
                                                            &storage->coord,
                                                            orient,
                                                            p);
    *ELEMENT = ( offset >= 0 ) ? 
                    ( (orient == mpi_matrix_orient_conj_transpose) ? conj(STORAGE->elements[offset]) : STORAGE->elements[offset] ) :
                    CMPLX(0.0, 0.0);
    return true;
}

bool
__mpi_matrix_storage_basic_complex_dp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) ) {
        mpi_matrix_storage_basic_complex_dp_t   *STORAGE = (mpi_matrix_storage_basic_complex_dp_t*)storage;
        double complex                          *ELEMENT = (double complex*)element;
        base_int_t                              offset = mpi_matrix_coord_index_to_offset(
                                                                &storage->coord,
                                                                orient,
                                                                p);
        if ( offset >= 0 ) {
            STORAGE->elements[offset] = *ELEMENT;
            return true;
        }
    }
    return false;
}

bool
__mpi_matrix_storage_basic_complex_dp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    double complex          zero = CMPLX(0.0, 0.0);
    return __mpi_matrix_storage_basic_complex_dp_set(storage, orient, p, &zero);
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_basic_complex_dp_callbacks = {
        NULL,
        __mpi_matrix_storage_basic_complex_dp_byte_usage,
        __mpi_matrix_storage_basic_complex_dp_clear,
        __mpi_matrix_storage_basic_complex_dp_get,
        __mpi_matrix_storage_basic_complex_dp_set
    };

//
////
//

static mpi_matrix_storage_callbacks_t* __mpi_matrix_storage_basic_callbacks_by_datatype[] = {
                    &__mpi_matrix_storage_basic_real_sp_callbacks,
                    &__mpi_matrix_storage_basic_real_dp_callbacks,
                    &__mpi_matrix_storage_basic_complex_sp_callbacks,
                    &__mpi_matrix_storage_basic_complex_dp_callbacks,
                    NULL
                };
static size_t __mpi_matrix_storage_basic_base_byte_sizes[] = {
                    sizeof(mpi_matrix_storage_basic_real_sp_t),
                    sizeof(mpi_matrix_storage_basic_real_dp_t),
                    sizeof(mpi_matrix_storage_basic_complex_sp_t),
                    sizeof(mpi_matrix_storage_basic_complex_dp_t),
                    0
                };

mpi_matrix_storage_ptr
__mpi_matrix_storage_basic_create(
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord
)
{
    size_t                  alloc_size = __mpi_matrix_storage_basic_base_byte_sizes[datatype]; 
    size_t                  data_size = mpi_matrix_storage_datatype_byte_sizes[datatype] * mpi_matrix_coord_element_count(coord);
    mpi_matrix_storage_t   *new_storage = (mpi_matrix_storage_t*)malloc(alloc_size + data_size);
    
    if ( new_storage ) {
        memset(new_storage, 0, alloc_size);
        new_storage->type = mpi_matrix_storage_type_basic;
        new_storage->datatype = datatype;
        new_storage->coord = *coord;
        new_storage->callbacks = *__mpi_matrix_storage_basic_callbacks_by_datatype[datatype];
        switch ( datatype ) {
            case mpi_matrix_storage_datatype_real_sp: {
                mpi_matrix_storage_basic_real_sp_t* STORAGE = (mpi_matrix_storage_basic_real_sp_t*)new_storage;
                STORAGE->elements = (void*)new_storage + data_size;
                break;
            }
            case mpi_matrix_storage_datatype_real_dp: {
                mpi_matrix_storage_basic_real_dp_t* STORAGE = (mpi_matrix_storage_basic_real_dp_t*)new_storage;
                STORAGE->elements = (void*)new_storage + data_size;
                break;
            }
            case mpi_matrix_storage_datatype_complex_sp: {
                mpi_matrix_storage_basic_complex_sp_t* STORAGE = (mpi_matrix_storage_basic_complex_sp_t*)new_storage;
                STORAGE->elements = (void*)new_storage + data_size;
                break;
            }
            case mpi_matrix_storage_datatype_complex_dp: {
                mpi_matrix_storage_basic_complex_dp_t* STORAGE = (mpi_matrix_storage_basic_complex_dp_t*)new_storage;
                STORAGE->elements = (void*)new_storage + data_size;
                break;
            }
            case mpi_matrix_storage_datatype_max:
        }
    }
    return new_storage;
}

mpi_matrix_storage_ptr
__mpi_matrix_storage_basic_create_immutable(
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord,
    bool                            should_alloc_array
)
{
    mpi_matrix_storage_t   *new_storage = NULL;
    
    if ( should_alloc_array ) {
        new_storage = (mpi_matrix_storage_t*)__mpi_matrix_storage_basic_create(datatype, coord);
        new_storage->options = mpi_matrix_storage_options_is_immutable;
    } else {
        new_storage = (mpi_matrix_storage_t*)malloc(__mpi_matrix_storage_basic_base_byte_sizes[datatype]);
        if ( new_storage ) {
            memset(new_storage, 0, __mpi_matrix_storage_basic_base_byte_sizes[datatype]);
            new_storage->type = mpi_matrix_storage_type_basic;
            new_storage->datatype = datatype;
            new_storage->options = mpi_matrix_storage_options_is_immutable;
            new_storage->coord = *coord;
            new_storage->callbacks = *__mpi_matrix_storage_basic_callbacks_by_datatype[datatype];
        }
    }
    return new_storage;
}

//
////
//

#ifndef MPI_MATRIX_STORAGE_SPARSE_PAGE_SIZE
#define MPI_MATRIX_STORAGE_SPARSE_PAGE_SIZE 4096
#endif

typedef struct mpi_matrix_storage_sparse_bst_minor_tuple {
    struct mpi_matrix_storage_sparse_bst_minor_tuple    *left, *right;      /* 2 pointers */
    base_int_t                                          j;                  /* 1 integer */
} mpi_matrix_storage_sparse_bst_minor_tuple_t;

/*
 * get the pointer to the attached storage in the minor tuple:
 */
#define mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(M)    ((void*)M + sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_t))

//

typedef struct mpi_matrix_storage_sparse_bst_minor_tuple_pool {
    struct mpi_matrix_storage_sparse_bst_minor_tuple_pool   *next_pool;                         /* 1 pointer */
    mpi_matrix_storage_sparse_bst_minor_tuple_t             *list_start, *list_end, *free_list; /* 3 pointers */
    unsigned char                                           tuple_storage[0];
} mpi_matrix_storage_sparse_bst_minor_tuple_pool_t;

//

static inline size_t
__mpi_matrix_storage_sparse_bst_minor_tuple_pool_byte_size(
    mpi_matrix_storage_datatype_t   datatype
)
{
    size_t              element_size = mpi_matrix_storage_datatype_byte_sizes[datatype];
    base_int_t          tuple_count = (MPI_MATRIX_STORAGE_SPARSE_PAGE_SIZE - sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_pool_t)) /
                                            (sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_t) + element_size);
    
    return sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_pool_t) + tuple_count * (sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_t) + element_size);
}

//

mpi_matrix_storage_sparse_bst_minor_tuple_pool_t*
__mpi_matrix_storage_sparse_bst_minor_tuple_pool_create(
    mpi_matrix_storage_datatype_t   datatype
)
{
    size_t                                              alloc_bytes = sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_pool_t);
    size_t                                              tuple_bytes;
    size_t                                              element_size = mpi_matrix_storage_datatype_byte_sizes[datatype];
    base_int_t                                          tuple_count;
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *new_pool = NULL;
    
    /*
     * How many tuples w/ value can we fit in the page?
     */
    tuple_count = (MPI_MATRIX_STORAGE_SPARSE_PAGE_SIZE - alloc_bytes) / (sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_t) + element_size);
    tuple_bytes = sizeof(mpi_matrix_storage_sparse_bst_minor_tuple_t) + element_size;
    alloc_bytes += tuple_count * tuple_bytes;
    
    /*
     * Allocate the pool:
     */
    new_pool = (mpi_matrix_storage_sparse_bst_minor_tuple_pool_t*)malloc(alloc_bytes);
    if ( new_pool ) {
        int         i;
        void        *tuple_storage = &new_pool->tuple_storage[0];
        
        memset(new_pool, 0, alloc_bytes);
        new_pool->list_start = new_pool->free_list = (mpi_matrix_storage_sparse_bst_minor_tuple_t*)tuple_storage;
        new_pool->list_end = (mpi_matrix_storage_sparse_bst_minor_tuple_t*)(tuple_storage + (tuple_count * tuple_bytes));
        i = 0;
        while ( i < tuple_count - 2 ) {
            ((mpi_matrix_storage_sparse_bst_minor_tuple_t*)tuple_storage)->right = (mpi_matrix_storage_sparse_bst_minor_tuple_t*)(tuple_storage + tuple_bytes);
            tuple_storage += tuple_bytes;
            i++;
        }
    }
    return new_pool;
}

//

mpi_matrix_storage_sparse_bst_minor_tuple_t*
__mpi_matrix_storage_sparse_bst_minor_tuple_pool_alloc(
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    **pool,
    mpi_matrix_storage_datatype_t                       datatype
)
{
    mpi_matrix_storage_sparse_bst_minor_tuple_t         *next_tuple = NULL;
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *root_pool = *pool;
    
    while ( root_pool ) {
        if ( root_pool->free_list ) {
            next_tuple = root_pool->free_list;
            root_pool->free_list = root_pool->free_list->right;
            next_tuple->left = next_tuple->right = NULL;
            break;
        }
        root_pool = root_pool->next_pool;
    }
    if ( ! next_tuple ) {
        // Need to add another page of tuples:
        mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *new_pool = __mpi_matrix_storage_sparse_bst_minor_tuple_pool_create(datatype);
        
        if ( new_pool ) {
            new_pool->next_pool = *pool;
            *pool = new_pool;
            next_tuple = new_pool->free_list;
            new_pool->free_list = new_pool->free_list->right;
        }
    }
    return next_tuple;
}

//

void
__mpi_matrix_storage_sparse_bst_minor_tuple_pool_dealloc(
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *pool,
    mpi_matrix_storage_sparse_bst_minor_tuple_t         *tuple
)
{
    while ( pool ) {
        if ( tuple >= pool->list_start && tuple < pool->list_end ) {
            tuple->right = pool->free_list;
            pool->free_list = tuple;
            break;
        }
        pool = pool->next_pool;
    }
}

//

void
__mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(
    mpi_matrix_storage_sparse_bst_minor_tuple_t     *root,
    mpi_matrix_storage_sparse_bst_minor_tuple_t     *others
)
{
    if ( others ) {
        mpi_matrix_storage_sparse_bst_minor_tuple_t *root_copy = root;
        mpi_matrix_storage_sparse_bst_minor_tuple_t *others_right = others->right;
        mpi_matrix_storage_sparse_bst_minor_tuple_t *others_left = others->left;
        
        while ( root_copy ) {
            if ( root_copy->j > others->j ) {
                if ( root_copy->left && root_copy->left->j > others->j ) {
                    root_copy = root_copy->left;
                } else {
                    others->right = NULL;
                    others->left = root_copy->left;
                    root_copy->left = others;
                    break;
                }
            } else {
                if ( root_copy->right && root_copy->right->j < others->j ) {
                    root_copy = root_copy->right;
                } else {
                    others->left = NULL;
                    others->right = root_copy->right;
                    root_copy->right = others;
                    break;
                }
            }
        }
        if ( others_left ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(root, others_left);
        if ( others_right ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(root, others_right);
    }
}

//
////
//

typedef struct mpi_matrix_storage_sparse_bst_major_tuple {
    struct mpi_matrix_storage_sparse_bst_major_tuple    *left, *right;  /* 2 pointers */
    mpi_matrix_storage_sparse_bst_minor_tuple_t         *row_siblings;  /* 1 pointer */
    base_int_t                                          i;              /* 1 integer */
} mpi_matrix_storage_sparse_bst_major_tuple_t;

//

#define mpi_matrix_storage_sparse_bst_major_tuple_pool_tuple_count \
            ((MPI_MATRIX_STORAGE_SPARSE_PAGE_SIZE - 4 * sizeof(void*)) / sizeof(mpi_matrix_storage_sparse_bst_major_tuple_t))

//
            
typedef struct mpi_matrix_storage_sparse_bst_major_tuple_pool {
    struct mpi_matrix_storage_sparse_bst_major_tuple_pool   *next_pool;                         /* 1 pointer */
    mpi_matrix_storage_sparse_bst_major_tuple_t             *list_start, *list_end, *free_list; /* 3 pointers */
    mpi_matrix_storage_sparse_bst_major_tuple_t             tuples[mpi_matrix_storage_sparse_bst_major_tuple_pool_tuple_count];
} mpi_matrix_storage_sparse_bst_major_tuple_pool_t;

//

#define __mpi_matrix_storage_sparse_bst_major_tuple_pool_byte_size(D) sizeof(mpi_matrix_storage_sparse_bst_major_tuple_pool_t)

//

mpi_matrix_storage_sparse_bst_major_tuple_pool_t*
__mpi_matrix_storage_sparse_bst_major_tuple_pool_create(
    mpi_matrix_storage_datatype_t   datatype
)
{
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *new_pool = (mpi_matrix_storage_sparse_bst_major_tuple_pool_t*)malloc(sizeof(mpi_matrix_storage_sparse_bst_major_tuple_pool_t));
    if ( new_pool ) {
        int         i;
        
        memset(new_pool, 0, sizeof(mpi_matrix_storage_sparse_bst_major_tuple_pool_t));
        new_pool->list_start = new_pool->free_list = &new_pool->tuples[0];
        new_pool->list_end = new_pool->tuples + mpi_matrix_storage_sparse_bst_major_tuple_pool_tuple_count;
        i = 0;
        while ( i < mpi_matrix_storage_sparse_bst_major_tuple_pool_tuple_count - 2 ) {
            new_pool->tuples[i].right = &new_pool->tuples[i+1];
            i++;
        }
    }
    return new_pool;
}

//

mpi_matrix_storage_sparse_bst_major_tuple_t*
__mpi_matrix_storage_sparse_bst_major_tuple_pool_alloc(
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    **pool,
    mpi_matrix_storage_datatype_t                       datatype
)
{
    mpi_matrix_storage_sparse_bst_major_tuple_t         *next_tuple = NULL;
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *root_pool = *pool;
    
    while ( root_pool ) {
        if ( root_pool->free_list ) {
            next_tuple = root_pool->free_list;
            root_pool->free_list = root_pool->free_list->right;
            next_tuple->left = next_tuple->right = NULL;
            next_tuple->row_siblings = NULL;
            break;
        }
        root_pool = root_pool->next_pool;
    }
    if ( ! next_tuple ) {
        // Need to add another page of tuples:
        mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *new_pool = __mpi_matrix_storage_sparse_bst_major_tuple_pool_create(datatype);
        
        if ( new_pool ) {
            new_pool->next_pool = *pool;
            *pool = new_pool;
            next_tuple = new_pool->free_list;
            new_pool->free_list = new_pool->free_list->right;
        }
    }
    return next_tuple;
}

//

void
__mpi_matrix_storage_sparse_bst_major_tuple_pool_dealloc(
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *pool,
    mpi_matrix_storage_sparse_bst_major_tuple_t         *tuple
)
{
    while ( pool ) {
        if ( tuple >= pool->list_start && tuple < pool->list_end ) {
            tuple->right = pool->free_list;
            pool->free_list = tuple;
            break;
        }
        pool = pool->next_pool;
    }
}

//
////
//

typedef struct {
    mpi_matrix_storage_t                                base;
    //
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *major_tuple_pool;
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *minor_tuple_pool;
    //
    mpi_matrix_storage_sparse_bst_major_tuple_t         *elements;
} mpi_matrix_storage_sparse_bst_t;

//

void
__mpi_matrix_storage_sparse_bst_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_bst_t                     *STORAGE = (mpi_matrix_storage_sparse_bst_t*)storage;
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *major = STORAGE->major_tuple_pool;
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *minor = STORAGE->minor_tuple_pool;
    
    // Drop all major tuple pools:
    while ( major ) {
        mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *next = major->next_pool;   
        free((void*)major);
        major = next;
    }
    // Drop all minor tuple pools:
    while ( minor ) {
        mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *next = minor->next_pool;   
        free((void*)minor);
        minor = next;
    }
}

size_t
__mpi_matrix_storage_sparse_bst_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_bst_t                     *STORAGE = (mpi_matrix_storage_sparse_bst_t*)storage;
    size_t                                              total_bytes = sizeof(mpi_matrix_storage_sparse_bst_t);
    mpi_matrix_storage_sparse_bst_major_tuple_pool_t    *major = STORAGE->major_tuple_pool;
    mpi_matrix_storage_sparse_bst_minor_tuple_pool_t    *minor = STORAGE->minor_tuple_pool;
    
    // Tally-up each major tuple pool:
    while ( major ) {
        total_bytes += __mpi_matrix_storage_sparse_bst_major_tuple_pool_byte_size(storage->datatype);
        major = major->next_pool;
    }
    // Tally-up each minor tuple pool:
    while ( minor ) {
        total_bytes += __mpi_matrix_storage_sparse_bst_minor_tuple_pool_byte_size(storage->datatype);
        minor = minor->next_pool;
    }

    return total_bytes;
}

bool
__mpi_matrix_storage_sparse_bst_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_bst_t             *STORAGE = (mpi_matrix_storage_sparse_bst_t*)storage;
    mpi_matrix_storage_sparse_bst_major_tuple_t *major = STORAGE->elements;
    base_int_t                                  idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    // Search for the major index first:
    idx = (storage->coord.is_row_major) ? p.i : p.j;
    while ( major ) {
        if ( major->i == idx ) break;
        if ( major->i > idx ) major = major->left;
        else major = major->right;
    }
    if ( major ) {
        mpi_matrix_storage_sparse_bst_minor_tuple_t *minor = major->row_siblings, *last_minor = NULL;
        
        idx = (storage->coord.is_row_major) ? p.j : p.i;
        while ( minor ) {
            if ( minor->j == idx ) break;
            last_minor = minor;
            if ( minor->j > idx ) minor = minor->left;
            else minor = minor->right;
        }
        if ( minor ) {
            // We found the element at minor; now remove minor from the row_siblings sub-tree:
            if ( last_minor ) {
                // The minor is NOT the root of the tree, hooray!
                if ( last_minor->left == minor ) {
                    // The minor is the left child of last_minor.  If minor has:
                    //     - a right child: attach the right child to minor->right then insert
                    //                      minor's left subtree into it
                    //     - a left child:  attach the left child to minor->left then insert
                    //                      minor's right subtree into it
                    //     - no children:   remove minor from last_minor->left
                    if ( minor->right ) {
                        last_minor->left = minor->right;
                        if ( minor->left ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(minor->right, minor->left);
                    }
                    else if ( minor->left ) {
                        last_minor->left = minor->left;
                        if ( minor->right ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(minor->left, minor->right);
                    }
                    else {
                        last_minor->left = NULL;
                    }
                } else {
                    // The minor is the right child of last_minor.  If minor has:
                    //     - a left child:  attach the left child to minor->left then insert
                    //                      minor's right subtree into it
                    //     - a right child: attach the right child to minor->right then insert
                    //                      minor's left subtree into it
                    //     - no children:   remove minor from last_minor->right
                    if ( minor->left ) {
                        last_minor->right = minor->left;
                        if ( minor->right ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(minor->left, minor->right);
                    }
                    else if ( minor->right ) {
                        last_minor->right = minor->right;
                        if ( minor->left ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(minor->right, minor->left);
                    }
                    else {
                        last_minor->right = NULL;
                    }
                }
            } else {
                // The minor is the root of the tree, great...
                if ( minor->left ) {
                    major->row_siblings = minor->left;
                    if ( minor->right ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(minor->left, minor->right);
                }
                else if ( minor->right ) {
                    major->row_siblings = minor->right;
                    if ( minor->left ) __mpi_matrix_storage_sparse_bst_minor_tuple_insert_tuples(minor->right, minor->left);
                }
                else {
                    major->row_siblings = NULL;
                }
            }
            __mpi_matrix_storage_sparse_bst_minor_tuple_pool_dealloc(STORAGE->minor_tuple_pool, minor);
            return true;
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_bst_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_bst_t             *STORAGE = (mpi_matrix_storage_sparse_bst_t*)storage;
    mpi_matrix_storage_sparse_bst_major_tuple_t *major = STORAGE->elements;
    base_int_t                                  idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    // Search for the major index first:
    idx = (storage->coord.is_row_major) ? p.i : p.j;
    while ( major ) {
        if ( major->i == idx ) {
            mpi_matrix_storage_sparse_bst_minor_tuple_t *minor = major->row_siblings;
            
            idx = (storage->coord.is_row_major) ? p.j : p.i;
            while ( minor ) {
                if ( minor->j == idx ) {
                    if ( orient == mpi_matrix_orient_conj_transpose ) {
                        if ( storage->datatype == mpi_matrix_storage_datatype_complex_sp ) {
                            float complex   *ELEMENT = (float complex*)element,
                                            *ORIGINAL = (float complex*)mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(minor);
                            *ELEMENT = conjf(*ORIGINAL);
                            return true;
                        }
                        else if ( storage->datatype == mpi_matrix_storage_datatype_complex_dp ) {
                            double complex  *ELEMENT = (double complex*)element,
                                            *ORIGINAL = (double complex*)mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(minor);
                            *ELEMENT = conj(*ORIGINAL);
                            return true;
                        }
                    }
                    memcpy(
                        element,
                        mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(minor),
                        mpi_matrix_storage_datatype_byte_sizes[storage->datatype]);
                    return true;
                }
                if ( minor->j > idx ) minor = minor->left;
                else minor = minor->right;
            }
            break;
        }
        if ( major->i > idx ) major = major->left;
        else major = major->right;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_bst_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_bst_t             *STORAGE = (mpi_matrix_storage_sparse_bst_t*)storage;
    mpi_matrix_storage_sparse_bst_major_tuple_t *major = STORAGE->elements;
    base_int_t                                  idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    // Search for the major index first:
    idx = (storage->coord.is_row_major) ? p.i : p.j;
    while ( major ) {
        if ( major->i == idx ) break;
        
        if ( major->i > idx ) {
            if ( major->left && major->left->i >= idx ) {
                major = major->left;
            } else {
                // The new element belongs on a major tuple between major and major->left:
                mpi_matrix_storage_sparse_bst_major_tuple_t *new_major = __mpi_matrix_storage_sparse_bst_major_tuple_pool_alloc(&STORAGE->major_tuple_pool, storage->datatype);
                
                if ( ! new_major ) {
                    // Memory error:
                    fprintf(stderr, "%s:%d - unable to allocate another sparse major tuple\n", __FILE__, __LINE__);
                    return false;
                }
                new_major->left = major->left;
                major->left = new_major;
                new_major->right = NULL;
                new_major->row_siblings = NULL;
                new_major->i = idx;
                major = new_major;
                break;
            }
        }
        else {
            if ( major->right && major->right->i <= idx ) {
                major = major->right;
            } else {
                // The new element belongs on a major tuple between major and major->right:
                mpi_matrix_storage_sparse_bst_major_tuple_t *new_major = __mpi_matrix_storage_sparse_bst_major_tuple_pool_alloc(&STORAGE->major_tuple_pool, storage->datatype);
                
                if ( ! new_major ) {
                    // Memory error:
                    fprintf(stderr, "%s:%d - unable to allocate another sparse major tuple\n", __FILE__, __LINE__);
                    return false;
                }
                new_major->right = major->right;
                major->right = new_major;
                new_major->left = NULL;
                new_major->row_siblings = NULL;
                new_major->i = idx;
                major = new_major;
                break;
            }
        }
    }
    if ( major ) {
        mpi_matrix_storage_sparse_bst_minor_tuple_t *minor = major->row_siblings, *last_minor = NULL;
        
        idx = (storage->coord.is_row_major) ? p.j : p.i;
        if ( minor ) {
            while ( minor ) {
                if ( minor->j == idx ) {
                    // Modify existing value:
                    memcpy(
                        mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(minor),
                        element,
                        mpi_matrix_storage_datatype_byte_sizes[storage->datatype]);
                    return true;
                }
                // If the index lies to the left, check that tuple; if it is less than idx, 
                // then a new minor tuple belongs between minor and minor->left:
                if ( minor->j > idx ) {
                    if ( minor->left && minor->left->j >= idx ) {
                        minor = minor->left;
                    } else {
                        // Attach new minor tuple to the left of minor:
                        mpi_matrix_storage_sparse_bst_minor_tuple_t *new_minor = __mpi_matrix_storage_sparse_bst_minor_tuple_pool_alloc(&STORAGE->minor_tuple_pool, storage->datatype);
                    
                        if ( ! new_minor ) {
                            // Memory error:
                            fprintf(stderr, "%s:%d - unable to allocate another sparse minor tuple\n", __FILE__, __LINE__);
                            return false;
                        }
                        new_minor->right = NULL;
                        new_minor->left = minor->left;
                        new_minor->j = idx;
                        memcpy(
                            mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(new_minor),
                            element,
                            mpi_matrix_storage_datatype_byte_sizes[storage->datatype]);
                        minor->left = new_minor;
                        return true;
                    }
                }
                // If the index lies to the right, check that tuple; if it is greater than idx, 
                // then a new minor tuple belongs between minor and minor->right:
                else {
                    if ( minor->right && minor->right->j <= idx ) {
                        minor = minor->right;
                    } else {
                        // Attach new minor tuple to the right of minor:
                        mpi_matrix_storage_sparse_bst_minor_tuple_t *new_minor = __mpi_matrix_storage_sparse_bst_minor_tuple_pool_alloc(&STORAGE->minor_tuple_pool, storage->datatype);
                    
                        if ( ! new_minor ) {
                            // Memory error:
                            fprintf(stderr, "%s:%d - unable to allocate another sparse minor tuple\n", __FILE__, __LINE__);
                            return false;
                        }
                        new_minor->left = NULL;
                        new_minor->right = minor->right;
                        new_minor->j = idx;
                        memcpy(
                            mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(new_minor),
                            element,
                            mpi_matrix_storage_datatype_byte_sizes[storage->datatype]);
                        minor->right = new_minor;
                        return true;
                    }
                }
            }
        } else {
            // Add to empty sibling chain
            mpi_matrix_storage_sparse_bst_minor_tuple_t *new_minor = __mpi_matrix_storage_sparse_bst_minor_tuple_pool_alloc(&STORAGE->minor_tuple_pool, storage->datatype);
            
            if ( ! new_minor ) {
                // Memory error:
                fprintf(stderr, "%s:%d - unable to allocate another sparse minor tuple\n", __FILE__, __LINE__);
                return false;
            }
            new_minor->left = new_minor->right = NULL;
            new_minor->j = idx;
            memcpy(
                mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(new_minor),
                element,
                mpi_matrix_storage_datatype_byte_sizes[storage->datatype]);
            major->row_siblings = new_minor;
            return true;
        }
    }
    return false;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_bst_callbacks = {
        __mpi_matrix_storage_sparse_bst_destroy,
        __mpi_matrix_storage_sparse_bst_byte_usage,
        __mpi_matrix_storage_sparse_bst_clear,
        __mpi_matrix_storage_sparse_bst_get,
        __mpi_matrix_storage_sparse_bst_set
    };



mpi_matrix_storage_ptr
__mpi_matrix_storage_sparse_bst_create(
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord
)
{
    mpi_matrix_storage_sparse_bst_t     *new_storage = (mpi_matrix_storage_sparse_bst_t*)malloc(sizeof(mpi_matrix_storage_sparse_bst_t));
    
    if ( new_storage ) {
        new_storage->base.type = mpi_matrix_storage_type_sparse_bst;
        new_storage->base.datatype = datatype;
        new_storage->base.coord = *coord;
        new_storage->base.callbacks = __mpi_matrix_storage_sparse_bst_callbacks;
        
        new_storage->major_tuple_pool = __mpi_matrix_storage_sparse_bst_major_tuple_pool_create(datatype);
        new_storage->minor_tuple_pool = __mpi_matrix_storage_sparse_bst_minor_tuple_pool_create(datatype);
        
        // For the sake of balance, add a single major tuple right in the middle of
        // the matrix:
        new_storage->elements = __mpi_matrix_storage_sparse_bst_major_tuple_pool_alloc(&new_storage->major_tuple_pool, datatype);
        if ( ! new_storage->elements ) {
            mpi_matrix_storage_destroy((mpi_matrix_storage_ptr)new_storage);
            // Memory error:
            fprintf(stderr, "%s:%d - unable to allocate a sparse major tuple\n", __FILE__, __LINE__);
            return NULL;
        }
        new_storage->elements->left = new_storage->elements->right = NULL;
        new_storage->elements->row_siblings = NULL;
        new_storage->elements->i = (coord->is_row_major) ? coord->dimensions.i / 2 : coord->dimensions.j / 2;
    }
    return (mpi_matrix_storage_ptr)new_storage;
}

//
////
//

#ifndef MPI_MATRIX_STORAGE_SPARSE_COMPRESSED_GROWTH_FACTOR
#define MPI_MATRIX_STORAGE_SPARSE_COMPRESSED_GROWTH_FACTOR 256
#endif

typedef void (*mpi_matrix_storage_sparse_compressed_init_callback)(mpi_matrix_storage_ptr storage);

//
// The CSR/CSC format requires a fixed-length list of base indices in the primary dimension
// and growable secondary-dimension and value lists of equivalent length.  While it would be
// tempting to bundle the secondary-dimension index and value into pairs and have a single list,
// that's not the mainstream CSR/CSC format.
//
// We will, however, allocate the primary dimension indices list as part of the object.
// itself.
//
typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    float                   *values;
    base_int_t              *secondary_indices;
    //
    base_int_t              *primary_indices;
} mpi_matrix_storage_sparse_compressed_real_sp_t;

void
__mpi_matrix_storage_sparse_compressed_real_sp_init(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_real_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
    STORAGE->primary_indices = (base_int_t*)((void*)storage + sizeof(mpi_matrix_storage_sparse_compressed_real_sp_t));
}

bool
__mpi_matrix_storage_sparse_compressed_real_sp_grow(
    mpi_matrix_storage_sparse_compressed_real_sp_t *storage
)
{
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_STORAGE_SPARSE_COMPRESSED_GROWTH_FACTOR;
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(float) + sizeof(base_int_t)));
    
    if ( new_space ) {
        float           *new_values = (float*)new_space;
        base_int_t      *new_secondary_indices = new_space + new_capacity * sizeof(float);
        
        if ( storage->nvalues ) {
            base_int_t  *old_secondary_indices = new_space + storage->capacity * sizeof(float);
                            
            // Move the secondary indices:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow compressed sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_compressed_real_sp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_real_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
    
    if ( STORAGE->capacity > 0 ) free((void*)STORAGE->values);
}

size_t
__mpi_matrix_storage_sparse_compressed_real_sp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_real_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_compressed_real_sp_t) + STORAGE->capacity * (sizeof(float) + sizeof(base_int_t)) +
                sizeof(base_int_t) * (1 + (storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j));
}

bool
__mpi_matrix_storage_sparse_compressed_real_sp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_compressed_real_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!  We need to remove the value and secondary index at offset:
                    if ( offset + 1 < STORAGE->nvalues ) {
                        memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset+1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
                        memmove(&STORAGE->values[offset], &STORAGE->values[offset+1], sizeof(float) * (STORAGE->nvalues - offset - 1));
                    }
                    // We lost a value:
                    STORAGE->nvalues--;
                    
                    // Any trailing rows beyond primary_idx will need to be incremented:
                    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]--;
                    
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_real_sp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
    float                                           *ELEMENT = (float*)element;
    base_int_t                                      offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    *ELEMENT = STORAGE->values[offset];
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_real_sp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
    float                                           *ELEMENT = (float*)element;
    base_int_t                                      offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    base_int_t                                      insert_at = 0, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    if ( STORAGE->nvalues > 0 ) {
        // Use the primary index to find our column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            base_int_t                  last_offset = -1;
            
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            
            // Do the binary search on the secondary index and either setting the
            // value if the index already exists, or determining at what index we need
            // to insert a new value:
            while ( offset_lo <= offset_hi ) {
                base_int_t      new_offset;
        
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    STORAGE->values[offset] = *ELEMENT;
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                if ( offset_lo > offset_hi ) break;
                new_offset = (offset_hi + offset_lo) / 2;
                if ( new_offset == offset ) break;
                last_offset = offset;
                offset = new_offset;
            }
            // Upon exiting the loop, we'll have a position bracketed by last_offset
            // and offset; we need to insert at the offset which is > the incoming
            // secondary index:
            if ( last_offset < offset ) {
                // Test last_offset first:
                if ( STORAGE->secondary_indices[last_offset] > secondary_idx ) offset = last_offset;
            }
            else if ( last_offset > offset ) {
                // Test offset first:
                if ( STORAGE->secondary_indices[offset] < secondary_idx ) offset = last_offset;
            }
            insert_at = offset;
        } else {
            // It was an empty index set for this row, insert right where we're currently pointed
            insert_at = offset_lo;
        }
    }
    
    // insert_at will now point to the secondary index/value position at which
    // the new value must be inserted; if we need to grow those lists do so now:
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_compressed_real_sp_grow(STORAGE) ) return false;
    
    // if insert_at is less than nvalues, we will need to shift indices and values
    // up:
    if ( insert_at < STORAGE->nvalues ) {
        memmove(&STORAGE->values[insert_at+1], &STORAGE->values[insert_at], sizeof(float) * (STORAGE->nvalues - insert_at));
        memmove(&STORAGE->secondary_indices[insert_at+1], &STORAGE->secondary_indices[insert_at], sizeof(base_int_t) * (STORAGE->nvalues - insert_at));
    }
    
    // Fill-in the value and secondary index:
    STORAGE->values[insert_at] = *ELEMENT;
    STORAGE->secondary_indices[insert_at] = secondary_idx;
    
    // Increment the number of values:
    STORAGE->nvalues++;
    
    // Any trailing rows beyond primary_idx will need to be incremented:
    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]++;
    
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_compressed_real_sp_callbacks = {
        __mpi_matrix_storage_sparse_compressed_real_sp_destroy,
        __mpi_matrix_storage_sparse_compressed_real_sp_byte_usage,
        __mpi_matrix_storage_sparse_compressed_real_sp_clear,
        __mpi_matrix_storage_sparse_compressed_real_sp_get,
        __mpi_matrix_storage_sparse_compressed_real_sp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    double                  *values;
    base_int_t              *secondary_indices;
    //
    base_int_t              *primary_indices;
} mpi_matrix_storage_sparse_compressed_real_dp_t;

void
__mpi_matrix_storage_sparse_compressed_real_dp_init(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_real_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_dp_t*)storage;
    STORAGE->primary_indices = (base_int_t*)((void*)storage + sizeof(mpi_matrix_storage_sparse_compressed_real_dp_t));
}

bool
__mpi_matrix_storage_sparse_compressed_real_dp_grow(
    mpi_matrix_storage_sparse_compressed_real_dp_t *storage
)
{
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_STORAGE_SPARSE_COMPRESSED_GROWTH_FACTOR;
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(double) + sizeof(base_int_t)));
    
    if ( new_space ) {
        double          *new_values = (double*)new_space;
        base_int_t      *new_secondary_indices = new_space + new_capacity * sizeof(double);
        
        if ( storage->nvalues ) {
            base_int_t  *old_secondary_indices = new_space + storage->capacity * sizeof(double);
                            
            // Move the secondary indices:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow compressed sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_compressed_real_dp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_real_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_dp_t*)storage;
    
    if ( STORAGE->capacity > 0 ) free((void*)STORAGE->values);
}

size_t
__mpi_matrix_storage_sparse_compressed_real_dp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_real_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_dp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_compressed_real_dp_t) + STORAGE->capacity * (sizeof(double) + sizeof(base_int_t)) +
                sizeof(base_int_t) * (1 + (storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j));
}

bool
__mpi_matrix_storage_sparse_compressed_real_dp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_compressed_real_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_real_dp_t*)storage;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                base_int_t      new_offset;
        
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!  We need to remove the value and secondary index at offset:
                    if ( offset + 1 < STORAGE->nvalues ) {
                        memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset+1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
                        memmove(&STORAGE->values[offset], &STORAGE->values[offset+1], sizeof(double) * (STORAGE->nvalues - offset - 1));
                    }
                    // We lost a value:
                    STORAGE->nvalues--;
                    
                    // Any trailing rows beyond primary_idx will need to be decremented:
                    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]--;
                    
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_real_dp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_compressed_real_dp_t*)storage;
    double                                          *ELEMENT = (double*)element;
    base_int_t                                      offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    *ELEMENT = STORAGE->values[offset];
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_real_dp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_compressed_real_dp_t*)storage;
    double                                          *ELEMENT = (double*)element;
    base_int_t                                      offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    base_int_t                                      insert_at = 0, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    if ( STORAGE->nvalues > 0 ) {
        // Use the primary index to find our column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            base_int_t                  last_offset = -1;
            
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            
            // Do the binary search on the secondary index and either setting the
            // value if the index already exists, or determining at what index we need
            // to insert a new value:
            while ( offset_lo <= offset_hi ) {
                base_int_t      new_offset;
        
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    STORAGE->values[offset] = *ELEMENT;
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                if ( offset_lo > offset_hi ) break;
                new_offset = (offset_hi + offset_lo) / 2;
                if ( new_offset == offset ) break;
                last_offset = offset;
                offset = new_offset;
            }
            // Upon exiting the loop, we'll have a position bracketed by last_offset
            // and offset; we need to insert at the offset which is > the incoming
            // secondary index:
            if ( last_offset < offset ) {
                // Test last_offset first:
                if ( STORAGE->secondary_indices[last_offset] > secondary_idx ) offset = last_offset;
            }
            else if ( last_offset > offset ) {
                // Test offset first:
                if ( STORAGE->secondary_indices[offset] < secondary_idx ) offset = last_offset;
            }
            insert_at = offset;
        } else {
            // It was an empty index set for this row, insert right where we're currently pointed
            insert_at = offset_lo;
        }
    }
    
    // insert_at will now point to the secondary index/value position at which
    // the new value must be inserted; if we need to grow those lists do so now:
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_compressed_real_dp_grow(STORAGE) ) return false;
    
    // if insert_at is less than nvalues, we will need to shift indices and values
    // up:
    if ( insert_at < STORAGE->nvalues ) {
        memmove(&STORAGE->values[insert_at+1], &STORAGE->values[insert_at], sizeof(double) * (STORAGE->nvalues - insert_at));
        memmove(&STORAGE->secondary_indices[insert_at+1], &STORAGE->secondary_indices[insert_at], sizeof(base_int_t) * (STORAGE->nvalues - insert_at));
    }
    
    // Fill-in the value and secondary index:
    STORAGE->values[insert_at] = *ELEMENT;
    STORAGE->secondary_indices[insert_at] = secondary_idx;
    
    // Increment the number of values:
    STORAGE->nvalues++;
    
    // Any trailing rows beyond primary_idx will need to be incremented:
    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]++;
    
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_compressed_real_dp_callbacks = {
        __mpi_matrix_storage_sparse_compressed_real_dp_destroy,
        __mpi_matrix_storage_sparse_compressed_real_dp_byte_usage,
        __mpi_matrix_storage_sparse_compressed_real_dp_clear,
        __mpi_matrix_storage_sparse_compressed_real_dp_get,
        __mpi_matrix_storage_sparse_compressed_real_dp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    float complex           *values;
    base_int_t              *secondary_indices;
    //
    base_int_t              *primary_indices;
} mpi_matrix_storage_sparse_compressed_complex_sp_t;

void
__mpi_matrix_storage_sparse_compressed_complex_sp_init(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_complex_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_sp_t*)storage;
    STORAGE->primary_indices = (base_int_t*)((void*)storage + sizeof(mpi_matrix_storage_sparse_compressed_complex_sp_t));
}

bool
__mpi_matrix_storage_sparse_compressed_complex_sp_grow(
    mpi_matrix_storage_sparse_compressed_complex_sp_t *storage
)
{
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_STORAGE_SPARSE_COMPRESSED_GROWTH_FACTOR;
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(float complex) + sizeof(base_int_t)));
    
    if ( new_space ) {
        float  complex  *new_values = (float complex*)new_space;
        base_int_t      *new_secondary_indices = new_space + new_capacity * sizeof(float complex);
        
        if ( storage->nvalues ) {
            base_int_t  *old_secondary_indices = new_space + storage->capacity * sizeof(float complex);
                            
            // Move the secondary indices:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow compressed sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_compressed_complex_sp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_complex_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_sp_t*)storage;
    
    if ( STORAGE->capacity > 0 ) free((void*)STORAGE->values);
}

size_t
__mpi_matrix_storage_sparse_compressed_complex_sp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_complex_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_sp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_compressed_complex_sp_t) + STORAGE->capacity * (sizeof(float complex) + sizeof(base_int_t)) +
                sizeof(base_int_t) * (1 + (storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j));
}

bool
__mpi_matrix_storage_sparse_compressed_complex_sp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_compressed_complex_sp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_sp_t*)storage;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!  We need to remove the value and secondary index at offset:
                    if ( offset + 1 < STORAGE->nvalues ) {
                        memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset+1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
                        memmove(&STORAGE->values[offset], &STORAGE->values[offset+1], sizeof(float complex) * (STORAGE->nvalues - offset - 1));
                    }
                    // We lost a value:
                    STORAGE->nvalues--;
                    
                    // Any trailing rows beyond primary_idx will need to be incremented:
                    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]--;
                    
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_complex_sp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_complex_sp_t   *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_sp_t*)storage;
    float complex                                       *ELEMENT = (float complex*)element;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    *ELEMENT = (orient == mpi_matrix_orient_conj_transpose) ? conjf(STORAGE->values[offset]) : STORAGE->values[offset];
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_complex_sp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_complex_sp_t   *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_sp_t*)storage;
    float complex                                       *ELEMENT = (float complex*)element;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    base_int_t                                          insert_at = 0, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    if ( STORAGE->nvalues > 0 ) {
        // Use the primary index to find our column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            base_int_t                  last_offset = -1;
            
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            
            // Do the binary search on the secondary index and either setting the
            // value if the index already exists, or determining at what index we need
            // to insert a new value:
            while ( offset_lo <= offset_hi ) {
                base_int_t      new_offset;
        
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    STORAGE->values[offset] = *ELEMENT;
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                if ( offset_lo > offset_hi ) break;
                new_offset = (offset_hi + offset_lo) / 2;
                if ( new_offset == offset ) break;
                last_offset = offset;
                offset = new_offset;
            }
            // Upon exiting the loop, we'll have a position bracketed by last_offset
            // and offset; we need to insert at the offset which is > the incoming
            // secondary index:
            if ( last_offset < offset ) {
                // Test last_offset first:
                if ( STORAGE->secondary_indices[last_offset] > secondary_idx ) offset = last_offset;
            }
            else if ( last_offset > offset ) {
                // Test offset first:
                if ( STORAGE->secondary_indices[offset] < secondary_idx ) offset = last_offset;
            }
            insert_at = offset;
        } else {
            // It was an empty index set for this row, insert right where we're currently pointed
            insert_at = offset_lo;
        }
    }
    
    // insert_at will now point to the secondary index/value position at which
    // the new value must be inserted; if we need to grow those lists do so now:
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_compressed_complex_sp_grow(STORAGE) ) return false;
    
    // if insert_at is less than nvalues, we will need to shift indices and values
    // up:
    if ( insert_at < STORAGE->nvalues ) {
        memmove(&STORAGE->values[insert_at+1], &STORAGE->values[insert_at], sizeof(float complex) * (STORAGE->nvalues - insert_at));
        memmove(&STORAGE->secondary_indices[insert_at+1], &STORAGE->secondary_indices[insert_at], sizeof(base_int_t) * (STORAGE->nvalues - insert_at));
    }
    
    // Fill-in the value and secondary index:
    STORAGE->values[insert_at] = *ELEMENT;
    STORAGE->secondary_indices[insert_at] = secondary_idx;
    
    // Increment the number of values:
    STORAGE->nvalues++;
    
    // Any trailing rows beyond primary_idx will need to be incremented:
    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]++;
    
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_compressed_complex_sp_callbacks = {
        __mpi_matrix_storage_sparse_compressed_complex_sp_destroy,
        __mpi_matrix_storage_sparse_compressed_complex_sp_byte_usage,
        __mpi_matrix_storage_sparse_compressed_complex_sp_clear,
        __mpi_matrix_storage_sparse_compressed_complex_sp_get,
        __mpi_matrix_storage_sparse_compressed_complex_sp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    double complex          *values;
    base_int_t              *secondary_indices;
    //
    base_int_t              *primary_indices;
} mpi_matrix_storage_sparse_compressed_complex_dp_t;

void
__mpi_matrix_storage_sparse_compressed_complex_dp_init(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_complex_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_dp_t*)storage;
    STORAGE->primary_indices = (base_int_t*)((void*)storage + sizeof(mpi_matrix_storage_sparse_compressed_complex_dp_t));
}

bool
__mpi_matrix_storage_sparse_compressed_complex_dp_grow(
    mpi_matrix_storage_sparse_compressed_complex_dp_t *storage
)
{
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_STORAGE_SPARSE_COMPRESSED_GROWTH_FACTOR;
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(double complex) + sizeof(base_int_t)));
    
    if ( new_space ) {
        double  complex *new_values = (double complex*)new_space;
        base_int_t      *new_secondary_indices = new_space + new_capacity * sizeof(double complex);
        
        if ( storage->nvalues ) {
            base_int_t  *old_secondary_indices = new_space + storage->capacity * sizeof(double complex);
                            
            // Move the secondary indices:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow compressed sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_compressed_complex_dp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_complex_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_dp_t*)storage;
    
    if ( STORAGE->capacity > 0 ) free((void*)STORAGE->values);
}

size_t
__mpi_matrix_storage_sparse_compressed_complex_dp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_compressed_complex_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_dp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_compressed_complex_dp_t) + STORAGE->capacity * (sizeof(double complex) + sizeof(base_int_t)) +
                sizeof(base_int_t) * (1 + (storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j));
}

bool
__mpi_matrix_storage_sparse_compressed_complex_dp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_compressed_complex_dp_t *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_dp_t*)storage;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!  We need to remove the value and secondary index at offset:
                    if ( offset + 1 < STORAGE->nvalues ) {
                        memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset+1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
                        memmove(&STORAGE->values[offset], &STORAGE->values[offset+1], sizeof(double complex) * (STORAGE->nvalues - offset - 1));
                    }
                    // We lost a value:
                    STORAGE->nvalues--;
                    
                    // Any trailing rows beyond primary_idx will need to be incremented:
                    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]--;
                    
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_complex_dp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_complex_dp_t   *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_dp_t*)storage;
    double complex                                      *ELEMENT = (double complex*)element;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    if ( STORAGE->nvalues > 0 ) {
        primary_idx = storage->coord.is_row_major ? p.i : p.j;
        secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
        // Use the primary index to find out column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    *ELEMENT = (orient == mpi_matrix_orient_conj_transpose) ? conj(STORAGE->values[offset]) : STORAGE->values[offset];
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                offset = (offset_hi + offset_lo) / 2;
            }
        }
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_compressed_complex_dp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_compressed_complex_dp_t   *STORAGE = (mpi_matrix_storage_sparse_compressed_complex_dp_t*)storage;
    double complex                                      *ELEMENT = (double complex*)element;
    base_int_t                                          offset_lo, offset_hi, offset, primary_idx, secondary_idx;
    base_int_t                                          insert_at = 0, primary_idx_max;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    primary_idx_max = storage->coord.is_row_major ? storage->coord.dimensions.i : storage->coord.dimensions.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    if ( STORAGE->nvalues > 0 ) {
        // Use the primary index to find our column/value offsets:
        offset_lo = STORAGE->primary_indices[primary_idx];
        offset_hi = STORAGE->primary_indices[primary_idx+1] - 1;
        
        if ( offset_hi >= offset_lo ) {
            base_int_t                  last_offset = -1;
            
            // Now we do a binary search across those columns:
            offset = (offset_hi + offset_lo) / 2;
            
            // Do the binary search on the secondary index and either setting the
            // value if the index already exists, or determining at what index we need
            // to insert a new value:
            while ( offset_lo <= offset_hi ) {
                base_int_t      new_offset;
        
                if ( STORAGE->secondary_indices[offset] == secondary_idx ) {
                    // Found it!
                    STORAGE->values[offset] = *ELEMENT;
                    return true;
                }
                if ( STORAGE->secondary_indices[offset] > secondary_idx ) {
                    offset_hi = offset - 1;
                } else {
                    offset_lo = offset + 1;
                }
                if ( offset_lo > offset_hi ) break;
                new_offset = (offset_hi + offset_lo) / 2;
                if ( new_offset == offset ) break;
                last_offset = offset;
                offset = new_offset;
            }
            // Upon exiting the loop, we'll have a position bracketed by last_offset
            // and offset; we need to insert at the offset which is > the incoming
            // secondary index:
            if ( last_offset < offset ) {
                // Test last_offset first:
                if ( STORAGE->secondary_indices[last_offset] > secondary_idx ) offset = last_offset;
            }
            else if ( last_offset > offset ) {
                // Test offset first:
                if ( STORAGE->secondary_indices[offset] < secondary_idx ) offset = last_offset;
            }
            insert_at = offset;
        } else {
            // It was an empty index set for this row, insert right where we're currently pointed
            insert_at = offset_lo;
        }
    }
    
    // insert_at will now point to the secondary index/value position at which
    // the new value must be inserted; if we need to grow those lists do so now:
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_compressed_complex_dp_grow(STORAGE) ) return false;
    
    // if insert_at is less than nvalues, we will need to shift indices and values
    // up:
    if ( insert_at < STORAGE->nvalues ) {
        memmove(&STORAGE->values[insert_at+1], &STORAGE->values[insert_at], sizeof(double complex) * (STORAGE->nvalues - insert_at));
        memmove(&STORAGE->secondary_indices[insert_at+1], &STORAGE->secondary_indices[insert_at], sizeof(base_int_t) * (STORAGE->nvalues - insert_at));
    }
    
    // Fill-in the value and secondary index:
    STORAGE->values[insert_at] = *ELEMENT;
    STORAGE->secondary_indices[insert_at] = secondary_idx;
    
    // Increment the number of values:
    STORAGE->nvalues++;
    
    // Any trailing rows beyond primary_idx will need to be incremented:
    while ( ++primary_idx <= primary_idx_max ) STORAGE->primary_indices[primary_idx]++;
    
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_compressed_complex_dp_callbacks = {
        __mpi_matrix_storage_sparse_compressed_complex_dp_destroy,
        __mpi_matrix_storage_sparse_compressed_complex_dp_byte_usage,
        __mpi_matrix_storage_sparse_compressed_complex_dp_clear,
        __mpi_matrix_storage_sparse_compressed_complex_dp_get,
        __mpi_matrix_storage_sparse_compressed_complex_dp_set
    };

//
////
//

static mpi_matrix_storage_callbacks_t* __mpi_matrix_storage_sparse_compressed_callbacks_by_datatype[] = {
                    &__mpi_matrix_storage_sparse_compressed_real_sp_callbacks,
                    &__mpi_matrix_storage_sparse_compressed_real_dp_callbacks,
                    &__mpi_matrix_storage_sparse_compressed_complex_sp_callbacks,
                    &__mpi_matrix_storage_sparse_compressed_complex_dp_callbacks,
                    NULL
                };
static mpi_matrix_storage_sparse_compressed_init_callback __mpi_matrix_storage_sparse_compressed_init_callbacks_by_datatype[] = {
                __mpi_matrix_storage_sparse_compressed_real_sp_init,
                __mpi_matrix_storage_sparse_compressed_real_dp_init,
                __mpi_matrix_storage_sparse_compressed_complex_sp_init,
                __mpi_matrix_storage_sparse_compressed_complex_dp_init,
                NULL
            };            

static size_t __mpi_matrix_storage_sparse_compressed_base_byte_sizes[] = {
                    sizeof(mpi_matrix_storage_sparse_compressed_real_sp_t),
                    sizeof(mpi_matrix_storage_sparse_compressed_real_dp_t),
                    sizeof(mpi_matrix_storage_sparse_compressed_complex_sp_t),
                    sizeof(mpi_matrix_storage_sparse_compressed_complex_dp_t),
                    0
                };

mpi_matrix_storage_ptr
__mpi_matrix_storage_sparse_compressed_create(
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord
)
{
    mpi_matrix_storage_t   *new_storage = NULL;
    size_t                  alloc_size = __mpi_matrix_storage_sparse_compressed_base_byte_sizes[datatype];
    
    // Factor-in the storage for the primary indices:
    base_int_t              primary_idx_max = coord->is_row_major ? coord->dimensions.i : coord->dimensions.j;
    
    alloc_size += (primary_idx_max + 1) * mpi_matrix_storage_datatype_byte_sizes[datatype];
    
    new_storage = (mpi_matrix_storage_t*)malloc(alloc_size);
    if ( new_storage ) {
        memset(new_storage, 0, alloc_size);
        new_storage->type = mpi_matrix_storage_type_sparse_compressed;
        new_storage->datatype = datatype;
        new_storage->coord = *coord;
        new_storage->callbacks = *__mpi_matrix_storage_sparse_compressed_callbacks_by_datatype[datatype];
        __mpi_matrix_storage_sparse_compressed_init_callbacks_by_datatype[datatype](new_storage);
    }
    return new_storage;
}

//
////
//

#ifndef MPI_MATRIX_STORAGE_SPARSE_COORDINATE_GROWTH_FACTOR
#define MPI_MATRIX_STORAGE_SPARSE_COORDINATE_GROWTH_FACTOR 4096
#endif

//
// The COO format uses three growable listts: primary- and secondary-dimension indices and value.
// While it would be tempting to bundle three as a struct and have a single list, that's not
// the mainstream COO format.
//
typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    float                   *values;
    base_int_t              *primary_indices;
    base_int_t              *secondary_indices;
} mpi_matrix_storage_sparse_coordinate_real_sp_t;

bool
__mpi_matrix_storage_sparse_coordinate_real_sp_find_index(
    mpi_matrix_storage_sparse_coordinate_real_sp_t  *storage,
    base_int_t                                      primary_idx,
    base_int_t                                      secondary_idx,
    base_int_t                                      *out_offset
)
{
    bool                    rc = false;
    base_int_t              offset = 0;
    
    switch ( storage->nvalues ) {
        case 0:
            break;
        case 1:
            if ( storage->primary_indices[0] == primary_idx ) {
                if ( storage->secondary_indices[0] == secondary_idx ) rc = true;
                else if ( storage->secondary_indices[0] < secondary_idx ) offset = 1;
            }
            else if ( storage->primary_indices[0] < primary_idx ) offset = 1;
            break;
    
        default: {
            // There are at least two values in the COO already.  Time for a binary search.
            base_int_t      offset_lo = 0, offset_hi = storage->nvalues - 1, last_offset = -1;
            bool            found_it = false;
            
            offset = (offset_lo + offset_hi) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( storage->primary_indices[offset] == primary_idx ) {
                    found_it = true;
                    break;
                }
                if ( storage->primary_indices[offset] < primary_idx ) {
                    offset_lo = offset + 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_lo;
                        break;
                    }
                } else {
                    offset_hi = offset - 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_hi;
                        break;
                    }
                }
                offset = (offset_lo + offset_hi) / 2;
                if ( offset == last_offset ) break;
                last_offset = offset;
            }
            
            // found_it indicates we found the primary index, so now it's a matter of doing a
            // linear search either left or right of the offset:
            if ( found_it ) {
                // Is the secondary index correct?
                if ( storage->secondary_indices[offset] == secondary_idx ) {
                    rc = true;
                }
                else if ( storage->secondary_indices[offset] < secondary_idx ) {
                    while ( ++offset < storage->nvalues ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] > secondary_idx ) break;
                    }
                }
                else {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] < secondary_idx ) {
                            offset++;
                            break;
                        }
                    }
                    if ( offset < 0 ) offset = 0;
                }   
            } else {
                // The offset might be inside a range of values from which we need to escape:
                if ( storage->primary_indices[offset] < primary_idx && offset < storage->nvalues ) {
                    while ( ++offset < storage->nvalues ) if ( storage->primary_indices[offset] >= primary_idx ) break;
                }
                else if ( storage->primary_indices[offset] > primary_idx && offset > 0 ) {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] < primary_idx ) {
                            offset++;
                            break;
                        }
                    }
                }
            }
            break;
        }
    }
    *out_offset = offset;
    return rc;
}

bool
__mpi_matrix_storage_sparse_coordinate_real_sp_grow(
    mpi_matrix_storage_sparse_coordinate_real_sp_t *storage
)
{
#define MPI_MATRIX_DELTA    MPI_MATRIX_STORAGE_SPARSE_COORDINATE_GROWTH_FACTOR / (sizeof(float) + 2 * sizeof(base_int_t));
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_DELTA;
#undef MPI_MATRIX_DELTA
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(float) + 2 * sizeof(base_int_t)));
    
    if ( new_space ) {
        float           *new_values = (float*)new_space;
        base_int_t      *new_primary_indices = new_space + new_capacity * sizeof(float);
        base_int_t      *new_secondary_indices = new_space + new_capacity * (sizeof(float) + sizeof(base_int_t));
        
        if ( storage->nvalues ) {
            base_int_t  *old_primary_indices = new_space + storage->capacity * sizeof(float);
            base_int_t  *old_secondary_indices = new_space + storage->capacity * (sizeof(float) + sizeof(base_int_t));
                            
            // Move the secondary indices first:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
            
            // Now that secondary is out of the way, move the primary indices:
            memmove(new_primary_indices, old_primary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->primary_indices = new_primary_indices;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow coordinate sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_coordinate_real_sp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_real_sp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)storage;
    
    if ( (storage->options & mpi_matrix_storage_options_is_mmap) && (STORAGE->capacity > 0) ) {
        munmap(STORAGE->values, STORAGE->nvalues * (sizeof(float) + 2 * sizeof(base_int_t)));
    }
    else if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) && (STORAGE->capacity > 0) ) {
        free((void*)STORAGE->values);
    }
}

size_t
__mpi_matrix_storage_sparse_coordinate_real_sp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_real_sp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_coordinate_real_sp_t) + STORAGE->capacity * (sizeof(float) + 2 * sizeof(base_int_t));
}

bool
__mpi_matrix_storage_sparse_coordinate_real_sp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)storage;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_real_sp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        // Delete at offset:
        if ( offset + 1 < STORAGE->nvalues ) {
            memmove(&STORAGE->values[offset], &STORAGE->values[offset + 1], sizeof(float) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->primary_indices[offset], &STORAGE->primary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
        }
        STORAGE->nvalues--;
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_real_sp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)storage;
    float                                           *ELEMENT = (float*)element;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_real_sp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        *ELEMENT = STORAGE->values[offset];
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_real_sp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)storage;
    float                                           *ELEMENT = (float*)element;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_real_sp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        STORAGE->values[offset] = *ELEMENT;
        return true;
    }
    
    // Are we at capacity?
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_coordinate_real_sp_grow(STORAGE) ) return false;
        
    // Insert the new indices and value at offset:
    if ( offset < STORAGE->nvalues ) {
        memmove(&STORAGE->values[offset+1], &STORAGE->values[offset], sizeof(float) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->primary_indices[offset+1], &STORAGE->primary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->secondary_indices[offset+1], &STORAGE->secondary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
    }
    STORAGE->values[offset] = *ELEMENT;
    STORAGE->primary_indices[offset] = primary_idx;
    STORAGE->secondary_indices[offset] = secondary_idx;
    STORAGE->nvalues++;
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_coordinate_real_sp_callbacks = {
        __mpi_matrix_storage_sparse_coordinate_real_sp_destroy,
        __mpi_matrix_storage_sparse_coordinate_real_sp_byte_usage,
        __mpi_matrix_storage_sparse_coordinate_real_sp_clear,
        __mpi_matrix_storage_sparse_coordinate_real_sp_get,
        __mpi_matrix_storage_sparse_coordinate_real_sp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    double                  *values;
    base_int_t              *primary_indices;
    base_int_t              *secondary_indices;
} mpi_matrix_storage_sparse_coordinate_real_dp_t;

bool
__mpi_matrix_storage_sparse_coordinate_real_dp_find_index(
    mpi_matrix_storage_sparse_coordinate_real_dp_t  *storage,
    base_int_t                                      primary_idx,
    base_int_t                                      secondary_idx,
    base_int_t                                      *out_offset
)
{
    bool                    rc = false;
    base_int_t              offset = 0;
    
    switch ( storage->nvalues ) {
        case 0:
            break;
        case 1:
            if ( storage->primary_indices[0] == primary_idx ) {
                if ( storage->secondary_indices[0] == secondary_idx ) rc = true;
                else if ( storage->secondary_indices[0] < secondary_idx ) offset = 1;
            }
            else if ( storage->primary_indices[0] < primary_idx ) offset = 1;
            break;
    
        default: {
            // There are at least two values in the COO already.  Time for a binary search.
            base_int_t      offset_lo = 0, offset_hi = storage->nvalues - 1, last_offset = -1;
            bool            found_it = false;
            
            offset = (offset_lo + offset_hi) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( storage->primary_indices[offset] == primary_idx ) {
                    found_it = true;
                    break;
                }
                if ( storage->primary_indices[offset] < primary_idx ) {
                    offset_lo = offset + 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_lo;
                        break;
                    }
                } else {
                    offset_hi = offset - 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_hi;
                        break;
                    }
                }
                offset = (offset_lo + offset_hi) / 2;
                if ( offset == last_offset ) break;
                last_offset = offset;
            }
            
            // found_it indicates we found the primary index, so now it's a matter of doing a
            // linear search either left or right of the offset:
            if ( found_it ) {
                // Is the secondary index correct?
                if ( storage->secondary_indices[offset] == secondary_idx ) {
                    rc = true;
                }
                else if ( storage->secondary_indices[offset] < secondary_idx ) {
                    while ( ++offset < storage->nvalues ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] > secondary_idx ) break;
                    }
                }
                else {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] < secondary_idx ) {
                            offset++;
                            break;
                        }
                    }
                    if ( offset < 0 ) offset = 0;
                }   
            } else {
                // The offset might be inside a range of values from which we need to escape:
                if ( storage->primary_indices[offset] < primary_idx && offset < storage->nvalues ) {
                    while ( ++offset < storage->nvalues ) if ( storage->primary_indices[offset] >= primary_idx ) break;
                }
                else if ( storage->primary_indices[offset] > primary_idx && offset > 0 ) {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] < primary_idx ) {
                            offset++;
                            break;
                        }
                    }
                }
            }
            break;
        }
    }
    *out_offset = offset;
    return rc;
}

bool
__mpi_matrix_storage_sparse_coordinate_real_dp_grow(
    mpi_matrix_storage_sparse_coordinate_real_dp_t *storage
)
{
#define MPI_MATRIX_DELTA    MPI_MATRIX_STORAGE_SPARSE_COORDINATE_GROWTH_FACTOR / (sizeof(double) + 2 * sizeof(base_int_t));
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_DELTA;
#undef MPI_MATRIX_DELTA
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(double) + 2 * sizeof(base_int_t)));
    
    if ( new_space ) {
        double          *new_values = (double*)new_space;
        base_int_t      *new_primary_indices = new_space + new_capacity * sizeof(double);
        base_int_t      *new_secondary_indices = new_space + new_capacity * (sizeof(double) + sizeof(base_int_t));
        
        if ( storage->nvalues ) {
            base_int_t  *old_primary_indices = new_space + storage->capacity * sizeof(double);
            base_int_t  *old_secondary_indices = new_space + storage->capacity * (sizeof(double) + sizeof(base_int_t));
                            
            // Move the secondary indices first:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
            
            // Now that secondary is out of the way, move the primary indices:
            memmove(new_primary_indices, old_primary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->primary_indices = new_primary_indices;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow coordinate sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_coordinate_real_dp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_real_dp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)storage;
    
    if ( (storage->options & mpi_matrix_storage_options_is_mmap) && (STORAGE->capacity > 0) ) {
        munmap(STORAGE->values, STORAGE->nvalues * (sizeof(double) + 2 * sizeof(base_int_t)));
    }
    else if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) && (STORAGE->capacity > 0) ) {
        free((void*)STORAGE->values);
    }
}

size_t
__mpi_matrix_storage_sparse_coordinate_real_dp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_real_dp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_coordinate_real_dp_t) + STORAGE->capacity * (sizeof(double) + 2 * sizeof(base_int_t));
}

bool
__mpi_matrix_storage_sparse_coordinate_real_dp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)storage;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_real_dp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        // Delete at offset:
        if ( offset + 1 < STORAGE->nvalues ) {
            memmove(&STORAGE->values[offset], &STORAGE->values[offset + 1], sizeof(double) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->primary_indices[offset], &STORAGE->primary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
        }
        STORAGE->nvalues--;
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_real_dp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)storage;
    double                                          *ELEMENT = (double*)element;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_real_dp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        *ELEMENT = STORAGE->values[offset];
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_real_dp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)storage;
    double                                          *ELEMENT = (double*)element;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_real_dp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        STORAGE->values[offset] = *ELEMENT;
        return true;
    }
    
    // Are we at capacity?
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_coordinate_real_dp_grow(STORAGE) ) return false;
        
    // Insert the new indices and value at offset:
    if ( offset < STORAGE->nvalues ) {
        memmove(&STORAGE->values[offset+1], &STORAGE->values[offset], sizeof(double) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->primary_indices[offset+1], &STORAGE->primary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->secondary_indices[offset+1], &STORAGE->secondary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
    }
    STORAGE->values[offset] = *ELEMENT;
    STORAGE->primary_indices[offset] = primary_idx;
    STORAGE->secondary_indices[offset] = secondary_idx;
    STORAGE->nvalues++;
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_coordinate_real_dp_callbacks = {
        __mpi_matrix_storage_sparse_coordinate_real_dp_destroy,
        __mpi_matrix_storage_sparse_coordinate_real_dp_byte_usage,
        __mpi_matrix_storage_sparse_coordinate_real_dp_clear,
        __mpi_matrix_storage_sparse_coordinate_real_dp_get,
        __mpi_matrix_storage_sparse_coordinate_real_dp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    float complex           *values;
    base_int_t              *primary_indices;
    base_int_t              *secondary_indices;
} mpi_matrix_storage_sparse_coordinate_complex_sp_t;

bool
__mpi_matrix_storage_sparse_coordinate_complex_sp_find_index(
    mpi_matrix_storage_sparse_coordinate_complex_sp_t  *storage,
    base_int_t                                      primary_idx,
    base_int_t                                      secondary_idx,
    base_int_t                                      *out_offset
)
{
    bool                    rc = false;
    base_int_t              offset = 0;
    
    switch ( storage->nvalues ) {
        case 0:
            break;
        case 1:
            if ( storage->primary_indices[0] == primary_idx ) {
                if ( storage->secondary_indices[0] == secondary_idx ) rc = true;
                else if ( storage->secondary_indices[0] < secondary_idx ) offset = 1;
            }
            else if ( storage->primary_indices[0] < primary_idx ) offset = 1;
            break;
    
        default: {
            // There are at least two values in the COO already.  Time for a binary search.
            base_int_t      offset_lo = 0, offset_hi = storage->nvalues - 1, last_offset = -1;
            bool            found_it = false;
            
            offset = (offset_lo + offset_hi) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( storage->primary_indices[offset] == primary_idx ) {
                    found_it = true;
                    break;
                }
                if ( storage->primary_indices[offset] < primary_idx ) {
                    offset_lo = offset + 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_lo;
                        break;
                    }
                } else {
                    offset_hi = offset - 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_hi;
                        break;
                    }
                }
                offset = (offset_lo + offset_hi) / 2;
                if ( offset == last_offset ) break;
                last_offset = offset;
            }
            
            // found_it indicates we found the primary index, so now it's a matter of doing a
            // linear search either left or right of the offset:
            if ( found_it ) {
                // Is the secondary index correct?
                if ( storage->secondary_indices[offset] == secondary_idx ) {
                    rc = true;
                }
                else if ( storage->secondary_indices[offset] < secondary_idx ) {
                    while ( ++offset < storage->nvalues ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] > secondary_idx ) break;
                    }
                }
                else {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] < secondary_idx ) {
                            offset++;
                            break;
                        }
                    }
                    if ( offset < 0 ) offset = 0;
                }   
            } else {
                // The offset might be inside a range of values from which we need to escape:
                if ( storage->primary_indices[offset] < primary_idx && offset < storage->nvalues ) {
                    while ( ++offset < storage->nvalues ) if ( storage->primary_indices[offset] >= primary_idx ) break;
                }
                else if ( storage->primary_indices[offset] > primary_idx && offset > 0 ) {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] < primary_idx ) {
                            offset++;
                            break;
                        }
                    }
                }
            }
            break;
        }
    }
    *out_offset = offset;
    return rc;
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_sp_grow(
    mpi_matrix_storage_sparse_coordinate_complex_sp_t *storage
)
{
#define MPI_MATRIX_DELTA    MPI_MATRIX_STORAGE_SPARSE_COORDINATE_GROWTH_FACTOR / (sizeof(float complex) + 2 * sizeof(base_int_t));
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_DELTA;
#undef MPI_MATRIX_DELTA
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(float complex) + 2 * sizeof(base_int_t)));
    
    if ( new_space ) {
        float complex   *new_values = (float complex*)new_space;
        base_int_t      *new_primary_indices = new_space + new_capacity * sizeof(float complex);
        base_int_t      *new_secondary_indices = new_space + new_capacity * (sizeof(float complex) + sizeof(base_int_t));
        
        if ( storage->nvalues ) {
            base_int_t  *old_primary_indices = new_space + storage->capacity * sizeof(float complex);
            base_int_t  *old_secondary_indices = new_space + storage->capacity * (sizeof(float complex) + sizeof(base_int_t));
                            
            // Move the secondary indices first:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
            
            // Now that secondary is out of the way, move the primary indices:
            memmove(new_primary_indices, old_primary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->primary_indices = new_primary_indices;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow coordinate sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_coordinate_complex_sp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_complex_sp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)storage;
    
    if ( (storage->options & mpi_matrix_storage_options_is_mmap) && (STORAGE->capacity > 0) ) {
        munmap(STORAGE->values, STORAGE->nvalues * (sizeof(complex float) + 2 * sizeof(base_int_t)));
    }
    else if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) && (STORAGE->capacity > 0) ) {
        free((void*)STORAGE->values);
    }
}

size_t
__mpi_matrix_storage_sparse_coordinate_complex_sp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_complex_sp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_coordinate_complex_sp_t) + STORAGE->capacity * (sizeof(float complex) + 2 * sizeof(base_int_t));
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_sp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_coordinate_complex_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)storage;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_complex_sp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        // Delete at offset:
        if ( offset + 1 < STORAGE->nvalues ) {
            memmove(&STORAGE->values[offset], &STORAGE->values[offset + 1], sizeof(float complex) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->primary_indices[offset], &STORAGE->primary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
        }
        STORAGE->nvalues--;
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_sp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_complex_sp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)storage;
    float complex                                       *ELEMENT = (float complex*)element;
    base_int_t                                          primary_idx, secondary_idx, offset;
    bool                                                extant_index;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_complex_sp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        *ELEMENT = (orient == mpi_matrix_orient_conj_transpose) ? conjf(STORAGE->values[offset]) : STORAGE->values[offset];
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_sp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_complex_sp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)storage;
    float complex                                       *ELEMENT = (float complex*)element;
    base_int_t                                          primary_idx, secondary_idx, offset;
    bool                                                extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_complex_sp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        STORAGE->values[offset] = *ELEMENT;
        return true;
    }
    
    // Are we at capacity?
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_coordinate_complex_sp_grow(STORAGE) ) return false;
        
    // Insert the new indices and value at offset:
    if ( offset < STORAGE->nvalues ) {
        memmove(&STORAGE->values[offset+1], &STORAGE->values[offset], sizeof(float complex) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->primary_indices[offset+1], &STORAGE->primary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->secondary_indices[offset+1], &STORAGE->secondary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
    }
    STORAGE->values[offset] = *ELEMENT;
    STORAGE->primary_indices[offset] = primary_idx;
    STORAGE->secondary_indices[offset] = secondary_idx;
    STORAGE->nvalues++;
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_coordinate_complex_sp_callbacks = {
        __mpi_matrix_storage_sparse_coordinate_complex_sp_destroy,
        __mpi_matrix_storage_sparse_coordinate_complex_sp_byte_usage,
        __mpi_matrix_storage_sparse_coordinate_complex_sp_clear,
        __mpi_matrix_storage_sparse_coordinate_complex_sp_get,
        __mpi_matrix_storage_sparse_coordinate_complex_sp_set
    };

//
////
//

typedef struct {
    mpi_matrix_storage_t    base;
    //
    base_int_t              nvalues, capacity;
    double complex          *values;
    base_int_t              *primary_indices;
    base_int_t              *secondary_indices;
} mpi_matrix_storage_sparse_coordinate_complex_dp_t;

bool
__mpi_matrix_storage_sparse_coordinate_complex_dp_find_index(
    mpi_matrix_storage_sparse_coordinate_complex_dp_t  *storage,
    base_int_t                                      primary_idx,
    base_int_t                                      secondary_idx,
    base_int_t                                      *out_offset
)
{
    bool                    rc = false;
    base_int_t              offset = 0;
    
    switch ( storage->nvalues ) {
        case 0:
            break;
        case 1:
            if ( storage->primary_indices[0] == primary_idx ) {
                if ( storage->secondary_indices[0] == secondary_idx ) rc = true;
                else if ( storage->secondary_indices[0] < secondary_idx ) offset = 1;
            }
            else if ( storage->primary_indices[0] < primary_idx ) offset = 1;
            break;
    
        default: {
            // There are at least two values in the COO already.  Time for a binary search.
            base_int_t      offset_lo = 0, offset_hi = storage->nvalues - 1, last_offset = -1;
            bool            found_it = false;
            
            offset = (offset_lo + offset_hi) / 2;
            while ( offset_lo <= offset_hi ) {
                if ( storage->primary_indices[offset] == primary_idx ) {
                    found_it = true;
                    break;
                }
                if ( storage->primary_indices[offset] < primary_idx ) {
                    offset_lo = offset + 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_lo;
                        break;
                    }
                } else {
                    offset_hi = offset - 1;
                    if ( offset_hi < offset_lo ) {
                        offset = offset_hi;
                        break;
                    }
                }
                offset = (offset_lo + offset_hi) / 2;
                if ( offset == last_offset ) break;
                last_offset = offset;
            }
            
            // found_it indicates we found the primary index, so now it's a matter of doing a
            // linear search either left or right of the offset:
            if ( found_it ) {
                // Is the secondary index correct?
                if ( storage->secondary_indices[offset] == secondary_idx ) {
                    rc = true;
                }
                else if ( storage->secondary_indices[offset] < secondary_idx ) {
                    while ( ++offset < storage->nvalues ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] > secondary_idx ) break;
                    }
                }
                else {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] != primary_idx ) break;
                        if ( storage->secondary_indices[offset] == secondary_idx ) {
                            rc = true;
                            break;
                        }
                        if ( storage->secondary_indices[offset] < secondary_idx ) {
                            offset++;
                            break;
                        }
                    }
                    if ( offset < 0 ) offset = 0;
                }   
            } else {
                // The offset might be inside a range of values from which we need to escape:
                if ( storage->primary_indices[offset] < primary_idx && offset < storage->nvalues ) {
                    while ( ++offset < storage->nvalues ) if ( storage->primary_indices[offset] >= primary_idx ) break;
                }
                else if ( storage->primary_indices[offset] > primary_idx && offset > 0 ) {
                    while ( offset-- > 0 ) {
                        if ( storage->primary_indices[offset] < primary_idx ) {
                            offset++;
                            break;
                        }
                    }
                }
            }
            break;
        }
    }
    *out_offset = offset;
    return rc;
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_dp_grow(
    mpi_matrix_storage_sparse_coordinate_complex_dp_t *storage
)
{
#define MPI_MATRIX_DELTA    MPI_MATRIX_STORAGE_SPARSE_COORDINATE_GROWTH_FACTOR / (sizeof(double complex) + 2 * sizeof(base_int_t));
    base_int_t          new_capacity = storage->capacity + MPI_MATRIX_DELTA;
#undef MPI_MATRIX_DELTA
    void                *new_space = realloc(storage->values, new_capacity * (sizeof(double complex) + 2 * sizeof(base_int_t)));
    
    if ( new_space ) {
        double complex  *new_values = (double complex*)new_space;
        base_int_t      *new_primary_indices = new_space + new_capacity * sizeof(double complex);
        base_int_t      *new_secondary_indices = new_space + new_capacity * (sizeof(double complex) + sizeof(base_int_t));
        
        if ( storage->nvalues ) {
            base_int_t  *old_primary_indices = new_space + storage->capacity * sizeof(double complex);
            base_int_t  *old_secondary_indices = new_space + storage->capacity * (sizeof(double complex) + sizeof(base_int_t));
                            
            // Move the secondary indices first:
            memmove(new_secondary_indices, old_secondary_indices, storage->nvalues * sizeof(base_int_t));
            
            // Now that secondary is out of the way, move the primary indices:
            memmove(new_primary_indices, old_primary_indices, storage->nvalues * sizeof(base_int_t));
        }
        storage->values = new_values;
        storage->primary_indices = new_primary_indices;
        storage->secondary_indices = new_secondary_indices;
        storage->capacity = new_capacity;
        return true;
    }
    fprintf(stderr, "%s:%d - unable to grow coordinate sparse lists\n", __FILE__, __LINE__);
    return false;
}

void
__mpi_matrix_storage_sparse_coordinate_complex_dp_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_complex_dp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)storage;
    
    if ( (storage->options & mpi_matrix_storage_options_is_mmap) && (STORAGE->capacity > 0) ) {
        munmap(STORAGE->values, STORAGE->nvalues * (sizeof(complex double) + 2 * sizeof(base_int_t)));
    }
    else if ( ! (storage->options & mpi_matrix_storage_options_is_immutable) && (STORAGE->capacity > 0) ) {
        free((void*)STORAGE->values);
    }
}

size_t
__mpi_matrix_storage_sparse_coordinate_complex_dp_byte_usage(
    mpi_matrix_storage_ptr  storage
)
{
    mpi_matrix_storage_sparse_coordinate_complex_dp_t *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)storage;
    
    return sizeof(mpi_matrix_storage_sparse_coordinate_complex_dp_t) + STORAGE->capacity * (sizeof(double complex) + 2 * sizeof(base_int_t));
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_dp_clear(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p
)
{
    mpi_matrix_storage_sparse_coordinate_complex_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)storage;
    base_int_t                                      primary_idx, secondary_idx, offset;
    bool                                            extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_complex_dp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        // Delete at offset:
        if ( offset + 1 < STORAGE->nvalues ) {
            memmove(&STORAGE->values[offset], &STORAGE->values[offset + 1], sizeof(double complex) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->primary_indices[offset], &STORAGE->primary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
            memmove(&STORAGE->secondary_indices[offset], &STORAGE->secondary_indices[offset + 1], sizeof(base_int_t) * (STORAGE->nvalues - offset - 1));
        }
        STORAGE->nvalues--;
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_dp_get(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_complex_dp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)storage;
    double complex                                      *ELEMENT = (double complex*)element;
    base_int_t                                          primary_idx, secondary_idx, offset;
    bool                                                extant_index;
    
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_complex_dp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        *ELEMENT = (orient == mpi_matrix_orient_conj_transpose) ? conj(STORAGE->values[offset]) : STORAGE->values[offset];
        return true;
    }
    return false;
}

bool
__mpi_matrix_storage_sparse_coordinate_complex_dp_set(
    mpi_matrix_storage_ptr  storage,
    mpi_matrix_orient_t     orient,
    int_pair_t              p,
    void                    *element
)
{
    mpi_matrix_storage_sparse_coordinate_complex_dp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)storage;
    double complex                                      *ELEMENT = (double complex*)element;
    base_int_t                                          primary_idx, secondary_idx, offset;
    bool                                                extant_index;
    
    if ( storage->options & mpi_matrix_storage_options_is_immutable ) return false;
    if ( ! mpi_matrix_coord_index_reduce(&storage->coord, orient, &p) ) return false;
    
    primary_idx = storage->coord.is_row_major ? p.i : p.j;
    secondary_idx = storage->coord.is_row_major ? p.j : p.i;
    
    extant_index = __mpi_matrix_storage_sparse_coordinate_complex_dp_find_index(STORAGE, primary_idx, secondary_idx, &offset);
    if ( extant_index ) {
        STORAGE->values[offset] = *ELEMENT;
        return true;
    }
    
    // Are we at capacity?
    if ( STORAGE->nvalues == STORAGE->capacity )
        if ( ! __mpi_matrix_storage_sparse_coordinate_complex_dp_grow(STORAGE) ) return false;
        
    // Insert the new indices and value at offset:
    if ( offset < STORAGE->nvalues ) {
        memmove(&STORAGE->values[offset+1], &STORAGE->values[offset], sizeof(double complex) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->primary_indices[offset+1], &STORAGE->primary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
        memmove(&STORAGE->secondary_indices[offset+1], &STORAGE->secondary_indices[offset], sizeof(base_int_t) * (STORAGE->nvalues - offset));
    }
    STORAGE->values[offset] = *ELEMENT;
    STORAGE->primary_indices[offset] = primary_idx;
    STORAGE->secondary_indices[offset] = secondary_idx;
    STORAGE->nvalues++;
    return true;
}

mpi_matrix_storage_callbacks_t __mpi_matrix_storage_sparse_coordinate_complex_dp_callbacks = {
        __mpi_matrix_storage_sparse_coordinate_complex_dp_destroy,
        __mpi_matrix_storage_sparse_coordinate_complex_dp_byte_usage,
        __mpi_matrix_storage_sparse_coordinate_complex_dp_clear,
        __mpi_matrix_storage_sparse_coordinate_complex_dp_get,
        __mpi_matrix_storage_sparse_coordinate_complex_dp_set
    };

//
////
//

static mpi_matrix_storage_callbacks_t* __mpi_matrix_storage_sparse_coordinate_callbacks_by_datatype[] = {
                    &__mpi_matrix_storage_sparse_coordinate_real_sp_callbacks,
                    &__mpi_matrix_storage_sparse_coordinate_real_dp_callbacks,
                    &__mpi_matrix_storage_sparse_coordinate_complex_sp_callbacks,
                    &__mpi_matrix_storage_sparse_coordinate_complex_dp_callbacks,
                    NULL
                };        

static size_t __mpi_matrix_storage_sparse_coordinate_base_byte_sizes[] = {
                    sizeof(mpi_matrix_storage_sparse_coordinate_real_sp_t),
                    sizeof(mpi_matrix_storage_sparse_coordinate_real_dp_t),
                    sizeof(mpi_matrix_storage_sparse_coordinate_complex_sp_t),
                    sizeof(mpi_matrix_storage_sparse_coordinate_complex_dp_t),
                    0
                };

mpi_matrix_storage_ptr
__mpi_matrix_storage_sparse_coordinate_create(
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord
)
{
    size_t                  alloc_size = __mpi_matrix_storage_sparse_coordinate_base_byte_sizes[datatype];
    mpi_matrix_storage_t   *new_storage = (mpi_matrix_storage_t*)malloc(alloc_size);
    
    if ( new_storage ) {
        memset(new_storage, 0, alloc_size);
        new_storage->type = mpi_matrix_storage_type_sparse_coordinate;
        new_storage->datatype = datatype;
        new_storage->coord = *coord;
        new_storage->callbacks = *__mpi_matrix_storage_sparse_coordinate_callbacks_by_datatype[datatype];
    }
    return new_storage;
}

mpi_matrix_storage_ptr
__mpi_matrix_storage_sparse_coordinate_create_immutable(
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord,
    base_int_t                      nvalues,
    bool                            should_alloc_arrays
)
{
    size_t                  addl_bytes = nvalues * (mpi_matrix_storage_datatype_byte_sizes[datatype] + 2 * sizeof(base_int_t));
    size_t                  alloc_size = __mpi_matrix_storage_sparse_coordinate_base_byte_sizes[datatype];
    mpi_matrix_storage_t   *new_storage = (mpi_matrix_storage_t*)malloc(alloc_size + ((should_alloc_arrays) ? addl_bytes : 0));
    
    if ( new_storage ) {
        void                *base_ptr = (void*)new_storage + alloc_size;
        
        memset(new_storage, 0, alloc_size);
        new_storage->type = mpi_matrix_storage_type_sparse_coordinate;
        new_storage->datatype = datatype;
        new_storage->coord = *coord;
        new_storage->options = mpi_matrix_storage_options_is_immutable;
        new_storage->callbacks = *__mpi_matrix_storage_sparse_coordinate_callbacks_by_datatype[datatype];
        
        switch ( datatype ) {
            case mpi_matrix_storage_datatype_real_sp: {
                mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)new_storage;
                
                STORAGE->nvalues = nvalues;
                STORAGE->capacity = nvalues;
                if ( should_alloc_arrays ) {
                    STORAGE->values = base_ptr; base_ptr += sizeof(float) * nvalues;
                    STORAGE->primary_indices = base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                    STORAGE->secondary_indices = base_ptr;
                } else {
                    STORAGE->values = NULL;
                    STORAGE->primary_indices = STORAGE->secondary_indices = NULL;
                }
                break;
            }
            case mpi_matrix_storage_datatype_real_dp: {
                mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)new_storage;
                
                STORAGE->nvalues = nvalues;
                STORAGE->capacity = nvalues;
                if ( should_alloc_arrays ) {
                    STORAGE->values = base_ptr; base_ptr += sizeof(double) * nvalues;
                    STORAGE->primary_indices = base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                    STORAGE->secondary_indices = base_ptr;
                } else {
                    STORAGE->values = NULL;
                    STORAGE->primary_indices = STORAGE->secondary_indices = NULL;
                }
                break;
            }
            case mpi_matrix_storage_datatype_complex_sp: {
                mpi_matrix_storage_sparse_coordinate_complex_sp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)new_storage;
                
                STORAGE->nvalues = nvalues;
                STORAGE->capacity = nvalues;
                if ( should_alloc_arrays ) {
                    STORAGE->values = base_ptr; base_ptr += sizeof(complex float) * nvalues;
                    STORAGE->primary_indices = base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                    STORAGE->secondary_indices = base_ptr;
                } else {
                    STORAGE->values = NULL;
                    STORAGE->primary_indices = STORAGE->secondary_indices = NULL;
                }
                break;
            }
            case mpi_matrix_storage_datatype_complex_dp: {
                mpi_matrix_storage_sparse_coordinate_complex_dp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)new_storage;
                
                STORAGE->nvalues = nvalues;
                STORAGE->capacity = nvalues;
                if ( should_alloc_arrays ) {
                    STORAGE->values = base_ptr; base_ptr += sizeof(complex double) * nvalues;
                    STORAGE->primary_indices = base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                    STORAGE->secondary_indices = base_ptr;
                } else {
                    STORAGE->values = NULL;
                    STORAGE->primary_indices = STORAGE->secondary_indices = NULL;
                }
                break;
            }
            case mpi_matrix_storage_datatype_max:
                break;
        }
    }
    return new_storage;
}

//
////
//

mpi_matrix_storage_ptr
mpi_matrix_storage_create(
    mpi_matrix_storage_type_t       type,
    mpi_matrix_storage_datatype_t   datatype,
    mpi_matrix_coord_ptr            coord
)
{
    mpi_matrix_storage_ptr          new_storage = NULL;
    
    if ( datatype >= 0 && datatype < mpi_matrix_storage_datatype_max ) {
        switch ( type ) {
            case mpi_matrix_storage_type_basic:
                new_storage = __mpi_matrix_storage_basic_create(datatype, coord);
                break;
            case mpi_matrix_storage_type_sparse_bst:
                new_storage = __mpi_matrix_storage_sparse_bst_create(datatype, coord);
                break;
            case mpi_matrix_storage_type_sparse_compressed:
                new_storage = __mpi_matrix_storage_sparse_compressed_create(datatype, coord);
                break;
            case mpi_matrix_storage_type_sparse_coordinate:
                new_storage = __mpi_matrix_storage_sparse_coordinate_create(datatype, coord);
                break;
            case mpi_matrix_storage_type_max:
                break;
        }
    }
    return new_storage;
}

//

void
mpi_matrix_storage_destroy(
    mpi_matrix_storage_ptr  storage
)
{
    if ( storage->callbacks.destroy ) storage->callbacks.destroy(storage);
    free((void*)storage);
}

//

bool
mpi_matrix_storage_basic_get_fields(
    mpi_matrix_storage_ptr  storage,
    int_pair_t              *dimensions,
    bool                    *is_row_major,
    base_int_t              *nvalues,
    const void*             *values
)
{
    if ( storage->type == mpi_matrix_storage_type_basic ) {
        mpi_matrix_storage_basic_real_sp_t  *STORAGE = (mpi_matrix_storage_basic_real_sp_t*)storage;
        
        if ( dimensions ) *dimensions = storage->coord.dimensions;
        if ( is_row_major ) *is_row_major = storage->coord.is_row_major;
        if ( nvalues ) *nvalues = mpi_matrix_coord_element_count(&storage->coord);
        if ( values ) *values = (void * const)STORAGE->elements;
        return true;
    }
    return false;
}

//

void
__mpi_matrix_storage_sparse_bst_to_compressed_minor_walk(
    mpi_matrix_storage_sparse_bst_minor_tuple_t *minor,
    base_int_t                                  primary_idx,
    mpi_matrix_storage_ptr                      out_storage
)
{
    int_pair_t      p = (out_storage->coord.is_row_major) ? int_pair_make(primary_idx, minor->j) : int_pair_make(minor->j, primary_idx);
    
    if ( minor->left ) __mpi_matrix_storage_sparse_bst_to_compressed_minor_walk(minor->left, primary_idx, out_storage);
    mpi_matrix_storage_set(out_storage, false, p, mpi_matrix_storage_sparse_bst_minor_tuple_element_ptr(minor));
    if ( minor->right ) __mpi_matrix_storage_sparse_bst_to_compressed_minor_walk(minor->right, primary_idx, out_storage);
}

void
__mpi_matrix_storage_sparse_bst_to_compressed_major_walk(
    mpi_matrix_storage_sparse_bst_major_tuple_t *major,
    mpi_matrix_storage_ptr                      out_storage
)
{
    if ( major->left ) __mpi_matrix_storage_sparse_bst_to_compressed_major_walk(major->left, out_storage);
    if ( major->row_siblings ) __mpi_matrix_storage_sparse_bst_to_compressed_minor_walk(major->row_siblings, major->i, out_storage);
    if ( major->right ) __mpi_matrix_storage_sparse_bst_to_compressed_major_walk(major->right, out_storage);
}

bool
mpi_matrix_storage_sparse_bst_to_compressed(
    mpi_matrix_storage_ptr      in_storage,
    mpi_matrix_storage_ptr      *out_storage
)
{
    if ( in_storage->type == mpi_matrix_storage_type_sparse_compressed ) {
        *out_storage = in_storage;
        return true;
    }
    if ( in_storage->type == mpi_matrix_storage_type_sparse_bst ) {
        mpi_matrix_storage_ptr  new_storage = mpi_matrix_storage_create(
                                                    mpi_matrix_storage_type_sparse_compressed,
                                                    in_storage->datatype,
                                                    &in_storage->coord
                                                );
        if ( new_storage ) {
            mpi_matrix_storage_sparse_bst_t     *STORAGE = (mpi_matrix_storage_sparse_bst_t*)in_storage;
            
            __mpi_matrix_storage_sparse_bst_to_compressed_major_walk(STORAGE->elements, new_storage);
            *out_storage = new_storage;
            return true;
        }
    }
    return false;
}

//

bool
mpi_matrix_storage_sparse_compressed_get_fields(
    mpi_matrix_storage_ptr      storage,
    int_pair_t                  *dimensions,
    bool                        *is_row_major,
    base_int_t                  *nvalues,
    const base_int_t*           *primary_indices,
    const base_int_t*           *secondary_indices,
    const void*                 *values
)
{
    if ( storage->type == mpi_matrix_storage_type_sparse_compressed ) {
        mpi_matrix_storage_sparse_compressed_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_compressed_real_sp_t*)storage;
        
        if ( dimensions ) *dimensions = storage->coord.dimensions;
        if ( is_row_major ) *is_row_major = storage->coord.is_row_major;
        if ( nvalues ) *nvalues = STORAGE->nvalues;
        if ( primary_indices ) *primary_indices = STORAGE->primary_indices;
        if ( secondary_indices ) *secondary_indices = STORAGE->secondary_indices;
        if ( values ) *values = (void * const)STORAGE->values;
        return true;
    }
    return false;
}

//

static uint64_t mpi_matrix_storage_sparse_coord_magic_header = 0x215852544d49504d;;
static uint32_t mpi_matrix_storage_sparse_coord_native_version = 0x00010000;

bool
__mpi_matrix_storage_write_header_to_fd(
    mpi_matrix_storage_ptr  storage,
    int                     fd,
    bool                    should_page_align,
    base_int_t              nvalues
)
{
#define DO_WRITE(P, S)      bytes_written = write(fd, (P), (S)); \
                            if ( bytes_written < (S) ) return false; else total_bytes_written += (S);

    uint8_t                 i8;
    uint32_t                i32;
    size_t                  total_bytes_written = 0;
    ssize_t                 bytes_written;
    
    DO_WRITE(&mpi_matrix_storage_sparse_coord_magic_header, sizeof(mpi_matrix_storage_sparse_coord_magic_header))
    DO_WRITE(&mpi_matrix_storage_sparse_coord_native_version, sizeof(mpi_matrix_storage_sparse_coord_native_version))

    i8 = (uint8_t)storage->type;
    DO_WRITE(&i8, sizeof(i8))

    i8 = (uint8_t)sizeof(base_int_t);
    DO_WRITE(&i8, sizeof(i8))

    i8 = (uint8_t)storage->datatype;
    DO_WRITE(&i8, sizeof(i8))

    i8 = (uint8_t)storage->coord.type;
    DO_WRITE(&i8, sizeof(i8))

    i8 = (uint8_t)storage->coord.is_row_major;
    DO_WRITE(&i8, sizeof(i8))

    i32 = should_page_align ? getpagesize() : 0;
    DO_WRITE(&i32, sizeof(i32))

    DO_WRITE(&storage->coord.dimensions.i, sizeof(storage->coord.dimensions.i))
    DO_WRITE(&storage->coord.dimensions.j, sizeof(storage->coord.dimensions.j))
    DO_WRITE(&storage->coord.k1, sizeof(storage->coord.k1))
    DO_WRITE(&storage->coord.k2, sizeof(storage->coord.k2))
    
    DO_WRITE(&nvalues, sizeof(nvalues))
    if ( should_page_align && ((getpagesize() - total_bytes_written) != 0) ) lseek(fd, (getpagesize() - total_bytes_written), SEEK_CUR);
    
    return true;

#undef DO_WRITE
}

bool
mpi_matrix_storage_write_to_fd(
    mpi_matrix_storage_ptr  storage,
    int                     fd,
    bool                    should_page_align
)
{
#define DO_WRITE(P, S)      bytes_written = write(fd, (P), (S)); \
                            if ( bytes_written < (S) ) return false; else total_bytes_written += (S);
                            
    size_t                  total_bytes_written = 0;
    ssize_t                 bytes_written;
    
    switch ( storage->type ) {
        case mpi_matrix_storage_type_basic: {
            base_int_t      nvalues = mpi_matrix_coord_element_count(&storage->coord);
            
            if ( __mpi_matrix_storage_write_header_to_fd(storage, fd, should_page_align, nvalues) ) {
                switch ( storage->datatype ) {
                    case mpi_matrix_storage_datatype_real_sp: {
                        mpi_matrix_storage_basic_real_sp_t* STORAGE = (mpi_matrix_storage_basic_real_sp_t*)storage;
                        DO_WRITE(STORAGE->elements, sizeof(float) * nvalues)
                        break;
                    }
                    case mpi_matrix_storage_datatype_real_dp: {
                        mpi_matrix_storage_basic_real_dp_t* STORAGE = (mpi_matrix_storage_basic_real_dp_t*)storage;
                        DO_WRITE(STORAGE->elements, sizeof(double) * nvalues)
                        break;
                    }
                    case mpi_matrix_storage_datatype_complex_sp: {
                        mpi_matrix_storage_basic_complex_sp_t* STORAGE = (mpi_matrix_storage_basic_complex_sp_t*)storage;
                        DO_WRITE(STORAGE->elements, sizeof(complex float) * nvalues)
                        break;
                    }
                    case mpi_matrix_storage_datatype_complex_dp: {
                        mpi_matrix_storage_basic_complex_dp_t* STORAGE = (mpi_matrix_storage_basic_complex_dp_t*)storage;
                        DO_WRITE(STORAGE->elements, sizeof(complex double) * nvalues)
                        break;
                    }
                    case mpi_matrix_storage_datatype_max:
                        break;
                }
                return true;
            }
            break;
        }
        case mpi_matrix_storage_type_sparse_coordinate: {
            switch ( storage->datatype ) {
                case mpi_matrix_storage_datatype_real_sp: {
                    mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = \
                            (mpi_matrix_storage_sparse_coordinate_real_sp_t*)storage;
                    
                    if ( __mpi_matrix_storage_write_header_to_fd(storage, fd, should_page_align, STORAGE->nvalues) ) {
                        DO_WRITE(STORAGE->values, STORAGE->nvalues * sizeof(float))
                        DO_WRITE(STORAGE->primary_indices, STORAGE->nvalues * sizeof(base_int_t))
                        DO_WRITE(STORAGE->secondary_indices, STORAGE->nvalues * sizeof(base_int_t))
                    }
                    break;
                }
                case mpi_matrix_storage_datatype_real_dp: {
                    mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = \
                            (mpi_matrix_storage_sparse_coordinate_real_dp_t*)storage;
                    
                    if ( __mpi_matrix_storage_write_header_to_fd(storage, fd, should_page_align, STORAGE->nvalues) ) {
                        DO_WRITE(STORAGE->values, STORAGE->nvalues * sizeof(double))
                        DO_WRITE(STORAGE->primary_indices, STORAGE->nvalues * sizeof(base_int_t))
                        DO_WRITE(STORAGE->secondary_indices, STORAGE->nvalues * sizeof(base_int_t))
                    }
                    break;
                }
                case mpi_matrix_storage_datatype_complex_sp: {
                    mpi_matrix_storage_sparse_coordinate_complex_sp_t  *STORAGE = \
                            (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)storage;
                    
                    if ( __mpi_matrix_storage_write_header_to_fd(storage, fd, should_page_align, STORAGE->nvalues) ) {
                        DO_WRITE(STORAGE->values, STORAGE->nvalues * sizeof(complex float))
                        DO_WRITE(STORAGE->primary_indices, STORAGE->nvalues * sizeof(base_int_t))
                        DO_WRITE(STORAGE->secondary_indices, STORAGE->nvalues * sizeof(base_int_t))
                    }
                    break;
                }
                case mpi_matrix_storage_datatype_complex_dp: {
                    mpi_matrix_storage_sparse_coordinate_complex_dp_t  *STORAGE = \
                            (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)storage;
                    
                    if ( __mpi_matrix_storage_write_header_to_fd(storage, fd, should_page_align, STORAGE->nvalues) ) {
                        DO_WRITE(STORAGE->values, STORAGE->nvalues * sizeof(complex double))
                        DO_WRITE(STORAGE->primary_indices, STORAGE->nvalues * sizeof(base_int_t))
                        DO_WRITE(STORAGE->secondary_indices, STORAGE->nvalues * sizeof(base_int_t))
                    }
                    break;
                }
                case mpi_matrix_storage_datatype_max:
                    break;
            }
            return true;
        }
        
        default:
            break;
    }
    return false;

#undef DO_WRITE
}

//

bool
__mpi_matrix_storage_read_header_from_fd(
    int                             fd,
    mpi_matrix_storage_type_t       *type,
    mpi_matrix_storage_datatype_t   *datatype,
    mpi_matrix_coord_t              *coord,
    base_int_t                      *nvalues,
    size_t                          *nbytes_read,
    const char*                     *error_msg
)
{
#define DO_READ(P, S)               bytes_read = read(fd, (P), (S)); \
                                    if ( bytes_read < (S) ) { \
                                        if ( error_msg ) *error_msg = "error while reading " #P; \
                                        return false; \
                                    } else total_bytes_read += bytes_read;
                            
    ssize_t                         bytes_read;
    size_t                          total_bytes_read = 0, page_size = 0;
    uint64_t                        i64;
    uint32_t                        i32;
    uint8_t                         i8;
    
    mpi_matrix_coord_type_t         coord_type;
    bool                            is_row_major;
    int_pair_t                      dimensions;
    base_int_t                      k1, k2;
    
    /*
     * Read and validate the file header:
     */
    DO_READ(&i64, sizeof(i64))
    if ( i64 != mpi_matrix_storage_sparse_coord_magic_header ) {
        if ( error_msg ) *error_msg = "invalid magic header";
        return false;
    }

    DO_READ(&i32, sizeof(i32))
    if ( i32 > mpi_matrix_storage_sparse_coord_native_version ) {
        if ( error_msg ) *error_msg = "file version newer than library version";
        return false;
    }

    DO_READ(&i8, sizeof(i8))
    if ( i8 >= mpi_matrix_storage_type_max ) {
        if ( error_msg ) *error_msg = "invalid matrix type from file";
        return false;
    }
    *type = i8;

    DO_READ(&i8, sizeof(i8))
    if ( i8 != sizeof(base_int_t) ) {
        if ( error_msg ) *error_msg = "integer size mismatch";
        return false;
    }
    
    DO_READ(&i8, sizeof(i8))
    if ( i8 >= mpi_matrix_storage_datatype_max ) {
        if ( error_msg ) *error_msg = "invalid datatype in file";
        return NULL;
    }
    *datatype = i8;
    
    DO_READ(&i8, sizeof(i8))
    if ( i8 >= mpi_matrix_coord_type_max ) {
        if ( error_msg ) *error_msg = "invalid coordinate type in file";
        return NULL;
    }
    coord_type = i8;
    
    DO_READ(&i8, sizeof(i8))
    if ( (i8 & ~(0x01)) ) {
        if ( error_msg ) *error_msg = "invalid row major bool in file";
        return NULL;
    }
    is_row_major = (i8 != 0);

    DO_READ(&i32, sizeof(i32))
    if ( i32 > 0 ) {
        if ( i32 != getpagesize() ) {
            if ( error_msg ) *error_msg = "page size mismatch between file and OS";
            return NULL;
        }
    }
    page_size = i32;
    
    DO_READ(&dimensions.i, sizeof(dimensions.i))
    DO_READ(&dimensions.j, sizeof(dimensions.j))
    DO_READ(&k1, sizeof(k1))
    DO_READ(&k2, sizeof(k2))
        
    switch ( coord_type ) {
        case mpi_matrix_coord_type_band_diagonal:
            mpi_matrix_coord_init(coord, coord_type, is_row_major, dimensions, k1, k2);
        default:
            mpi_matrix_coord_init(coord, coord_type, is_row_major, dimensions);
            break;
    }
        
    DO_READ(nvalues, sizeof(*nvalues))
    if ( page_size && (page_size - total_bytes_read != 0) ) total_bytes_read = lseek(fd, (page_size - total_bytes_read), SEEK_CUR);
    
    *nbytes_read = total_bytes_read;
    return true;
#undef DO_READ
}
    

mpi_matrix_storage_ptr
mpi_matrix_storage_read_from_fd(
    int                     fd,
    bool                    is_shared_memory,
    const char*             *error_msg
)
{
#define DO_READ(P, S)       bytes_read = read(fd, (P), (S)); \
                            if ( bytes_read < (S) ) { \
                                if ( error_msg ) *error_msg = "error while reading " #P; \
                                return NULL; \
                            } else total_bytes_read += bytes_read;
                            
    mpi_matrix_storage_ptr          new_storage = NULL;
    ssize_t                         bytes_read;
    size_t                          total_bytes_read = 0;
    mpi_matrix_storage_type_t       type;
    mpi_matrix_storage_datatype_t   datatype;
    mpi_matrix_coord_t              coord;
    base_int_t                      nvalues;
    
    /*
     * If we fail to read the file header, no sense continuing:
     */
    if ( ! __mpi_matrix_storage_read_header_from_fd(fd, &type, &datatype, &coord, &nvalues, &total_bytes_read, error_msg) ) return NULL;
    
    switch ( type ) {
        case mpi_matrix_storage_type_basic: {
            /*
             * We're ready to initialize the new storage object:
             */
            new_storage = __mpi_matrix_storage_basic_create_immutable(datatype, &coord, ! is_shared_memory);    
            if ( new_storage ) {
                if ( is_shared_memory ) {
                    void            *base_ptr = mmap(
                                                    NULL,
                                                    mpi_matrix_coord_element_count(&coord) * mpi_matrix_storage_datatype_byte_sizes[datatype],
                                                    PROT_READ,
                                                    MAP_SHARED,
                                                    fd,
                                                    total_bytes_read);
                    if ( base_ptr != MAP_FAILED ) {
                        switch ( datatype ) {
                            case mpi_matrix_storage_datatype_real_sp: {
                                mpi_matrix_storage_basic_real_sp_t* STORAGE = (mpi_matrix_storage_basic_real_sp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->elements = base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_real_dp: {
                                mpi_matrix_storage_basic_real_dp_t* STORAGE = (mpi_matrix_storage_basic_real_dp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->elements = base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_complex_sp: {
                                mpi_matrix_storage_basic_complex_sp_t* STORAGE = (mpi_matrix_storage_basic_complex_sp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->elements = base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_complex_dp: {
                                mpi_matrix_storage_basic_complex_dp_t* STORAGE = (mpi_matrix_storage_basic_complex_dp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->elements = base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_max:
                                break;
                        }
                    } else {
                        mpi_matrix_storage_destroy(new_storage);
                        new_storage = NULL;
                        if ( error_msg ) *error_msg = "unable to mmap shared memory region";
                    }
                } else {
                    switch ( datatype ) {
                        case mpi_matrix_storage_datatype_real_sp: {
                            mpi_matrix_storage_basic_real_sp_t* STORAGE = (mpi_matrix_storage_basic_real_sp_t*)new_storage;
                            DO_READ(STORAGE->elements, sizeof(float) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_real_dp: {
                            mpi_matrix_storage_basic_real_dp_t* STORAGE = (mpi_matrix_storage_basic_real_dp_t*)new_storage;
                            DO_READ(STORAGE->elements, sizeof(double) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_complex_sp: {
                            mpi_matrix_storage_basic_complex_sp_t* STORAGE = (mpi_matrix_storage_basic_complex_sp_t*)new_storage;
                            DO_READ(STORAGE->elements, sizeof(complex float) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_complex_dp: {
                            mpi_matrix_storage_basic_complex_dp_t* STORAGE = (mpi_matrix_storage_basic_complex_dp_t*)new_storage;
                            DO_READ(STORAGE->elements, sizeof(complex double) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_max:
                            break;
                    }
                }
            } else {
                if ( error_msg ) *error_msg = "unable to allocate mpi_matrix_storage object";
            }
            break;
        }
        
        case mpi_matrix_storage_type_sparse_coordinate: {
            /*
             * We're ready to initialize the new storage object:
             */
            new_storage = __mpi_matrix_storage_sparse_coordinate_create_immutable(datatype, &coord, nvalues, ! is_shared_memory);    
            if ( new_storage ) {
                if ( is_shared_memory ) {
                    void            *base_ptr = mmap(
                                                    NULL,
                                                    nvalues * (mpi_matrix_storage_datatype_byte_sizes[datatype] + 2 * sizeof(base_int_t)),
                                                    PROT_READ,
                                                    MAP_SHARED,
                                                    fd,
                                                    total_bytes_read);
                    if ( base_ptr != MAP_FAILED ) {
                        switch ( datatype ) {
                            case mpi_matrix_storage_datatype_real_sp: {
                                mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->values = (float*)base_ptr; base_ptr += sizeof(float) * nvalues;
                                STORAGE->primary_indices = (base_int_t*)base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                                STORAGE->secondary_indices = (base_int_t*)base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_real_dp: {
                                mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->values = (double*)base_ptr; base_ptr += sizeof(double) * nvalues;
                                STORAGE->primary_indices = (base_int_t*)base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                                STORAGE->secondary_indices = (base_int_t*)base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_complex_sp: {
                                mpi_matrix_storage_sparse_coordinate_complex_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->values = (complex float*)base_ptr; base_ptr += sizeof(complex float) * nvalues;
                                STORAGE->primary_indices = (base_int_t*)base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                                STORAGE->secondary_indices = (base_int_t*)base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_complex_dp: {
                                mpi_matrix_storage_sparse_coordinate_complex_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)new_storage;
                                STORAGE->base.options |= mpi_matrix_storage_options_is_mmap;
                                STORAGE->values = (complex double*)base_ptr; base_ptr += sizeof(complex double) * nvalues;
                                STORAGE->primary_indices = (base_int_t*)base_ptr; base_ptr += sizeof(base_int_t) * nvalues;
                                STORAGE->secondary_indices = (base_int_t*)base_ptr;
                                break;
                            }
                            case mpi_matrix_storage_datatype_max:
                                break;
                        }
                    } else {
                        mpi_matrix_storage_destroy(new_storage);
                        new_storage = NULL;
                        if ( error_msg ) *error_msg = "unable to mmap shared memory region";
                    }
                } else {
                    switch ( datatype ) {
                        case mpi_matrix_storage_datatype_real_sp: {
                            mpi_matrix_storage_sparse_coordinate_real_sp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_sp_t*)new_storage;
                    
                            DO_READ(STORAGE->values, sizeof(float) * nvalues)
                            DO_READ(STORAGE->primary_indices, sizeof(base_int_t) * nvalues)
                            DO_READ(STORAGE->secondary_indices, sizeof(base_int_t) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_real_dp: {
                            mpi_matrix_storage_sparse_coordinate_real_dp_t  *STORAGE = (mpi_matrix_storage_sparse_coordinate_real_dp_t*)new_storage;

                            DO_READ(STORAGE->values, sizeof(double) * nvalues)
                            DO_READ(STORAGE->primary_indices, sizeof(base_int_t) * nvalues)
                            DO_READ(STORAGE->secondary_indices, sizeof(base_int_t) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_complex_sp: {
                            mpi_matrix_storage_sparse_coordinate_complex_sp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_sp_t*)new_storage;
                
                            DO_READ(STORAGE->values, sizeof(complex float) * nvalues)
                            DO_READ(STORAGE->primary_indices, sizeof(base_int_t) * nvalues)
                            DO_READ(STORAGE->secondary_indices, sizeof(base_int_t) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_complex_dp: {
                            mpi_matrix_storage_sparse_coordinate_complex_dp_t   *STORAGE = (mpi_matrix_storage_sparse_coordinate_complex_dp_t*)new_storage;
                
                            DO_READ(STORAGE->values, sizeof(complex double) * nvalues)
                            DO_READ(STORAGE->primary_indices, sizeof(base_int_t) * nvalues)
                            DO_READ(STORAGE->secondary_indices, sizeof(base_int_t) * nvalues)
                            break;
                        }
                        case mpi_matrix_storage_datatype_max:
                            break;
            
                    }
                }
            } else {
                if ( error_msg ) *error_msg = "unable to allocate mpi_matrix_storage object";
            }
            break;
        }
        
        default: {
            if ( error_msg ) *error_msg = "invalid matrix type";
            break;
        }
    }
    return new_storage;
}
