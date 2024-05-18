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
typedef bool (*mpi_matrix_storage_clear_callback)(mpi_matrix_storage_ptr storage, bool is_transpose, int_pair_t p);

/*
 * @typedef mpi_matrix_storage_get_callback
 *
 * The type of a function that retrieves a single matrix element
 * at p.
 *
 * Returns boolean false if the value could not be retrieved, otherwise
 * boolean true and *element is set.
 */
typedef bool (*mpi_matrix_storage_get_callback)(mpi_matrix_storage_ptr storage, bool is_transpose, int_pair_t p, void *element);

/*
 * @typedef mpi_matrix_storage_set_callback
 *
 * The type of a function that stores a single matrix element value
 * at p.
 *
 * Returns boolean false if the value could not be set, otherwise
 * boolean true.
 */
typedef bool (*mpi_matrix_storage_set_callback)(mpi_matrix_storage_ptr storage, bool is_transpose, int_pair_t p, void *element);

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

#endif /* __MPI_MATRIX_COORD_H__ */
