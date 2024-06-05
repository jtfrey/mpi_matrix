/*	mpi_matrix_storage.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header MPI matrix storage pseudo-classes
	
	Pseudo-classes that handle the storage of matrix elements.
*/

#ifndef __MPI_MATRIX_STORAGE_H__
#define __MPI_MATRIX_STORAGE_H__

#include "mpi_matrix_coord.h"

/*
 * @typedef mpi_matrix_storage_type_t
 *
 * The type of matrix storage system:
 *
 *     basic: a linear array of elements of the matrix datatype is allocated
 *         according to the number of elements associated with the mpi_matrix_coord
 *
 *     sparse, bst: a binary search tree of primary indices has each node holding
 *         a binary search tree of secondary indices and values
 *
 *     sparse, compressed: Compressed Sparse Row (CSR) and Column (CSC) with the
 *         variant depending on the row- vs. column-major aspect of the
 *         mpi_matrix_coord
 *
 *     sparse, coordinate: COOrdinate sparse storage which stores the (i,j,value)
 *         tuples as independent lists sorted by primary index then secondary
 *         index (for row-major, i is the primary index); values are located
 *         using an augmented binary search on the primary index list and a
 *         subsequent directional scan of the secondary indices
 */
typedef enum {
    mpi_matrix_storage_type_basic = 0,
    mpi_matrix_storage_type_sparse_bst,
    mpi_matrix_storage_type_sparse_compressed,
    mpi_matrix_storage_type_sparse_coordinate,
    mpi_matrix_storage_type_max
} mpi_matrix_storage_type_t;

/*
 * @function mpi_matrix_storage_type_get_name
 *
 * Return a textual description of the matrix storage type.
 */
const char* mpi_matrix_storage_type_get_name(mpi_matrix_storage_type_t type);

/*
 * @typedef mpi_matrix_storage_datatype_t
 *
 * The data type of the matrix elements being stored.  The underlying
 * C types are:
 *
 *     float
 *     double
 *     float complex
 *     double complex
 */
typedef enum {
    mpi_matrix_storage_datatype_real_sp = 0,
    mpi_matrix_storage_datatype_real_dp,
    mpi_matrix_storage_datatype_complex_sp,
    mpi_matrix_storage_datatype_complex_dp,
    mpi_matrix_storage_datatype_max
} mpi_matrix_storage_datatype_t;

/*
 * @function mpi_matrix_storage_datatype_get_name
 *
 * Return a textual description of the matrix storage data type.
 */
const char* mpi_matrix_storage_datatype_get_name(mpi_matrix_storage_datatype_t type);

/*
 * @constant mpi_matrix_storage_datatype_byte_sizes
 *
 * Number of bytes associated with each datatype, ordered by the enum values
 * in mpi_matrix_storage_datatype_t:  the size of a real_dp is found at
 * mpi_matrix_storage_datatype_byte_sizes[mpi_matrix_storage_datatype_real_dp].
 */
extern const size_t mpi_matrix_storage_datatype_byte_sizes[];

/*
 * @function mpi_matrix_storage_datatype_get_byte_size_for_count
 *
 * Convenience function that computes the number of bytes required by count
 * elements of the given datatype.
 */
static inline size_t
mpi_matrix_storage_datatype_get_byte_size_for_count(
    mpi_matrix_storage_datatype_t   datatype,
    base_int_t                      count
)
{
    return mpi_matrix_storage_datatype_byte_sizes[datatype] * count;
}

/*
 * @typedef mpi_matrix_storage_ptr
 *
 * An opaque pointer to a mpi_matrix_storage instance.
 * Forward declaration for the sake of the callback function
 * typedefs.
 */
typedef struct mpi_matrix_storage const * mpi_matrix_storage_ptr;

/*
 * @typedef mpi_matrix_storage_destroy_callback
 *
 * The type of a function that destroys any dynamically-allocated
 * memory associated with storage.
 */
typedef void (*mpi_matrix_storage_destroy_callback)(mpi_matrix_storage_ptr storage);

/*
 * @typedef mpi_matrix_storage_byte_usage_callback
 *
 * The type of a function that checks how much memory storage is
 * currently using.
 */
typedef size_t (*mpi_matrix_storage_byte_usage_callback)(mpi_matrix_storage_ptr storage);

/*
 * @typedef mpi_matrix_storage_clear_callback
 *
 * The type of a function that "unsets" a single matrix element
 * at p.
 *
 * Returns boolean false if the value could not be cleared, true
 * otherwise.
 */
typedef bool (*mpi_matrix_storage_clear_callback)(mpi_matrix_storage_ptr storage, mpi_matrix_orient_t orient, int_pair_t p);

/*
 * @typedef mpi_matrix_storage_get_callback
 *
 * The type of a function that retrieves a single matrix element
 * at p.
 *
 * Returns boolean false if the value could not be retrieved, otherwise
 * boolean true and *element is set.
 */
typedef bool (*mpi_matrix_storage_get_callback)(mpi_matrix_storage_ptr storage, mpi_matrix_orient_t orient, int_pair_t p, void *element);

/*
 * @typedef mpi_matrix_storage_set_callback
 *
 * The type of a function that stores a single matrix element value
 * at p.
 *
 * Returns boolean false if the value could not be set, otherwise
 * boolean true.
 */
typedef bool (*mpi_matrix_storage_set_callback)(mpi_matrix_storage_ptr storage, mpi_matrix_orient_t orient, int_pair_t p, void *element);

/*
 * @typedef mpi_matrix_storage_callbacks_t
 *
 * The list of callback functions associated with a matrix
 * storage instance.
 */
typedef struct {
    mpi_matrix_storage_destroy_callback     destroy;
    mpi_matrix_storage_byte_usage_callback  byte_usage;
    mpi_matrix_storage_clear_callback       clear;
    mpi_matrix_storage_get_callback         get;
    mpi_matrix_storage_set_callback         set;
} mpi_matrix_storage_callbacks_t;

/*
 * @typedef mpi_matrix_storage_t
 *
 * The data structure of a matrix storage system.
 */
typedef struct mpi_matrix_storage {
    mpi_matrix_storage_type_t       type;
    mpi_matrix_storage_datatype_t   datatype;
    unsigned int                    options;
    mpi_matrix_coord_t              coord;
    mpi_matrix_storage_callbacks_t  callbacks;
} mpi_matrix_storage_t;

/*
 * @function mpi_matrix_storage_create
 *
 * Dynamically allocate a new matrix storage system of the given
 * type and dimensions.  The given coordinate system, coord,
 * is used to determine how large the storage must be and how
 * elements are addressed.
 *
 * Returns NULL if the instance could not be allocated.
 */
mpi_matrix_storage_ptr mpi_matrix_storage_create(mpi_matrix_storage_type_t type, mpi_matrix_storage_datatype_t datatype, mpi_matrix_coord_ptr coord);

/*
 * @function mpi_matrix_storage_destroy
 *
 * Destroy a dynamically-allocated instance of mpi_matrix_storage.
 */
void mpi_matrix_storage_destroy(mpi_matrix_storage_ptr storage);

/*
 * @defined mpi_matrix_storage_get_type_name
 *
 * Convenience wrapper to getting the textual description of
 * the instance's matrix storage type.
 */
#define mpi_matrix_storage_get_type_name(C) (mpi_matrix_storage_type_get_name((C)->type))

/*
 * @defined mpi_matrix_storage_get_datatype_name
 *
 * Convenience wrapper to getting the textual description of
 * the instance's fundamental datatype.
 */
#define mpi_matrix_storage_get_datatype_name(C) (mpi_matrix_storage_datatype_get_name((C)->datatype))

/*
 * @defined mpi_matrix_storage_byte_usage
 *
 * Convenience wrapper to the instance's byte_usage callback.
 */
#define mpi_matrix_storage_byte_usage(C) ((C)->callbacks.byte_usage((C)))

/*
 * @defined mpi_matrix_storage_clear
 *
 * Convenience wrapper to the instance's clear callback.
 */
#define mpi_matrix_storage_clear(C, T, P) ((C)->callbacks.clear((C), (T), (P)))

/*
 * @defined mpi_matrix_storage_get
 *
 * Convenience wrapper to the instance's get callback.
 */
#define mpi_matrix_storage_get(C, T, P, E) ((C)->callbacks.get((C), (T), (P), (E)))

/*
 * @defined mpi_matrix_storage_set
 *
 * Convenience wrapper to the instance's set callback.
 */
#define mpi_matrix_storage_set(C, T, P, E) ((C)->callbacks.set((C), (T), (P), (E)))

/*
 * @function mpi_matrix_storage_basic_get_fields
 *
 * Retrieve metadata for storage, which must be an instance of the
 * basic storage type.
 *
 * Pass NULL for any of the field arguments that are not required:
 *
 *   dimensions: the row, column size of storage
 *   is_row_major: true if storage orders values by row first
 *   nvalues: the number of matrix elements present in the flat list
 *   values: pointer to the flat list of matrix elements; the typeless
 *      pointer should be cast to float/double (real vs. complex) by
 *      the caller (exercise caution!!)
 *
 * Returns false if storage is NOT of the basic type.
 */
bool
mpi_matrix_storage_basic_get_fields(
    mpi_matrix_storage_ptr  storage,
    int_pair_t              *dimensions,
    bool                    *is_row_major,
    base_int_t              *nvalues,
    const void*             *values);

/*
 * @function mpi_matrix_storage_sparse_compressed_get_fields
 *
 * Retrieve metadata for storage, which must be an instance of the
 * sparse compressed storage type.
 *
 * Pass NULL for any of the field arguments that are not required:
 *
 *   dimensions: the row, column size of storage
 *   is_row_major: true if storage orders values by row first
 *   nvalues: the number of secondary indices and matrix elements present
 *      in the flat lists of secondary_indices and values
 *   primary_indices: pointer to the flat list of primary index offsets
 *      into the secondary_indices flat list; this list is dimensioned to
 *      the leading dimension of the matrix plus one
 *   secondary_indices: pointer to the flat list of secondary indices
 *      associated with values at the same offset; indices in each row
 *      are sorted in ascending order
 *   values: pointer to the flat list of matrix elements; the typeless
 *      pointer should be cast to float/double (real vs. complex) by
 *      the caller (exercise caution!!)
 *
 * Returns false if storage is NOT of the sparse compressed type.
 */
bool
mpi_matrix_storage_sparse_compressed_get_fields(
    mpi_matrix_storage_ptr  storage,
    int_pair_t              *dimensions,
    bool                    *is_row_major,
    base_int_t              *nvalues,
    const base_int_t*       *primary_indices,
    const base_int_t*       *secondary_indices,
    const void*             *values);

/*
 * @function mpi_matrix_storage_sparse_bst_to_compressed
 *
 * Convert a binary search tree sparse matrix to the equivalent matrix
 * in sparse compressed format.
 *
 * If successful, *out_storage is set to the new instance of
 * mpi_matrix_storage and true is returned.
 *
 * Returns false otherwise.
 */
bool
mpi_matrix_storage_sparse_bst_to_compressed(
    mpi_matrix_storage_ptr      in_storage,
    mpi_matrix_storage_ptr      *out_storage);

/*
 * @function mpi_matrix_storage_write_to_fd
 *
 * Given a matrix, storage, attempt to write the data present in it
 * to the given file descriptor, fd.  The file will contain data
 * in a layout that is specific to the storage type, with a common header.
 *
 * If the data in the file will be used to mmap() a shared memory segment
 * into multiple processes, the should_page_align flag must be true.  This
 * will ensure the payload (all after the header) is on a page boundary.
 *
 * Write is currently implemented for basic and sparse (COORDinate)
 * matrix storage types.
 *
 *     header section:
 *         uint64_t     magic_header = 0x215852544d49504d; //'MPIMTRX!' little endian
 *         uint32_t     version =      0x00010000;         // 1.0.0 [4B][2B][2B]
 *         uint8_t      subtype = mpi_matrix_storage_type_basic;
 *         uint8_t      intsize = (4|8);
 *         uint8_t      datatype = mpi_matrix_storage_datatype_real_dp;
 *         uint8_t      coord_type = mpi_matrix_coord_type_upper_triangular;
 *         uint8_t      is_row_major = (0|1);
 *         uint32_t     in_file_alignment = (0|<page size>);
 *         <int_type>   dim_i, dim_j, k1, k2;
 *         <int_type>   n_values;
 *           :
 *          byte padding to page boundary (optional)
 *           :
 *
 *       The padding is introduced if should_page_align is true.  This is primarily
 *       for the sake of mmap()'ing the file content via shared memory.
 *
 *     basic type:
 *                      ...header section...
 *         <datatype>   values[nvalues];
 *
 *     sparse, coordinate:
 *                      ...header section...
 *         <datatype>   values[n_values];
 *         <int_type>   primary_indices[n_values];
 *         <int_type>   secondary_indices[n_values];
 */
bool
mpi_matrix_storage_write_to_fd(
    mpi_matrix_storage_ptr  storage,
    int                     fd,
    bool                    should_page_align);

/*
 * @function mpi_matrix_storage_read_from_fd
 *
 * Attempt to allocate and initialize a new matrix from the data in the
 * given file descriptor, fd.
 *
 * If error_msg is non-NULL, *error_msg will be set to a string constant
 * on error and NULL will be returned.
 *
 */
mpi_matrix_storage_ptr
mpi_matrix_storage_read_from_fd(
    int                     fd,
    bool                    is_shared_memory,
    const char*             *error_msg);

#endif /* __MPI_MATRIX_COORD_H__ */
