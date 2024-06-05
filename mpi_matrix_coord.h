/*	mpi_matrix_coord.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header MPI matrix coordinate pseudo-classes
	
	Pseudo-classes that handle the internal coordinate system of
	a matrix in different storage formats.  E.g. a triangular
	matrix uses coordinate reflection to cut the required storage
	versus the corresponding full matrix.
	
	The API includes callbacks that will convert a (i,j) pair to
	a flat offset and to return the number of elements present
	in the matrix according to its type.
*/

#ifndef __MPI_MATRIX_COORD_H__
#define __MPI_MATRIX_COORD_H__

#include "int_pair.h"

/*
 * @typedef mpi_matrix_coord_type_t
 *
 * The type of matrix coordinate system.
 *
 * A full type defines a value for every (i,j) pair.
 * 
 * An upper-triangular type defines a value for all (i,j) pairs for
 * which j >= i.  The non-explicit pairs (j,i) are equivalent to (i,j).
 * 
 * A lower-triangular type defines a value for all (i,j) pairs for
 * which j <= i.  The non-explicit pairs (j,i) are equivalent to (i,j).
 *
 * A band diagonal type requires two additional configuration
 * parameters, k1 and k2, which are the number of additional diagonal
 * bands to the left and right of the main diagonal, respectively.
 * The tridiagonal (k1=k2=1) and diagonal (k1=k2=0) types are both
 * band diagonal, but they have explicit implementations since their
 * computations are far more straightforward.  Note, too, that upper-
 * and lower-triangular matrices are also technically band diagonal
 * with either k1 or k2 being zero.
 *
 * A tridiagonal type defines a value for all (i,j) pairs along the
 * primary diagonal and the two secondary diagonals.  All other pairs
 * do not exist (e.g. they are implicitly zero).
 *
 * A diagonal type defines a value for all (i,j) pairs along the
 * primary diagonal.  All other pairs do not exist (e.g. they are
 * implicitly zero).
 */
typedef enum {
    mpi_matrix_coord_type_full = 0,
    mpi_matrix_coord_type_upper_triangular,
    mpi_matrix_coord_type_lower_triangular,
    mpi_matrix_coord_type_band_diagonal,
    mpi_matrix_coord_type_tridiagonal,
    mpi_matrix_coord_type_diagonal,
    mpi_matrix_coord_type_max
} mpi_matrix_coord_type_t;

/*
 * @function mpi_matrix_coord_type_get_name
 *
 * Return a textual description of the matrix coordinate type.
 */
const char* mpi_matrix_coord_type_get_name(mpi_matrix_coord_type_t type);

/*
 * @enum MPI matrix coordinate status flags
 *
 * A bitmask that describes the nature of a coordinate index.
 *
 *     - is_defined: set if (i,j) is valid
 *     - is_unique:  set if the (i,j) does not map to a different
 *          (i',j') due to symmetry
 */
enum {
    mpi_matrix_coord_status_is_defined = 1 << 0,
    mpi_matrix_coord_status_is_unique = 1 << 1
};

/*
 * @typedef mpi_matrix_coord_status_t
 *
 * The type of a MPI matrix coordinate status bitmask.
 */
typedef unsigned int mpi_matrix_coord_status_t;

/*
 * @typedef mpi_matrix_orient_t
 *
 * Orientation of matrix coordinate relative to the underlying
 * matrix coordinate system.  Transpose implies (i,j) accesses
 * element (j,i), conjugate transpose accesses the complex
 * conjugate of (j,i).
 */
typedef enum {
    mpi_matrix_orient_normal = 0,
    mpi_matrix_orient_transpose = 't',
    mpi_matrix_orient_conj_transpose = 'c'
} mpi_matrix_orient_t;

/*
 * @typedef mpi_matrix_coord_ptr
 *
 * An opaque pointer to a mpi_matrix_coord instance.
 * Forward declaration for the sake of the callback function
 * typedefs.
 */
typedef struct mpi_matrix_coord const * mpi_matrix_coord_ptr;

/*
 * @typedef mpi_matrix_coord_element_count_callback
 *
 * The type of a function that returns the number of matrix elements
 * defined by a matrix coordinate system.
 */
typedef base_int_t (*mpi_matrix_coord_element_count_callback)(mpi_matrix_coord_ptr coord);

/*
 * @typedef mpi_matrix_coord_index_status
 *
 * Returns a coordinate status bitmask for p.
 */
typedef mpi_matrix_coord_status_t (*mpi_matrix_coord_index_status_callback)(mpi_matrix_coord_ptr coord, mpi_matrix_orient_t orient, int_pair_t p);

/*
 * @typedef mpi_matrix_coord_index_reduce
 *
 * Possibly alter the coordinate p to account for underlying matrix
 * symmetry.
 */
typedef bool (*mpi_matrix_coord_index_reduce_callback)(mpi_matrix_coord_ptr coord, mpi_matrix_orient_t orient, int_pair_t *p);

/*
 * @typedef mpi_matrix_coord_index_to_offset_callback
 *
 * The type of a function that converts the matrix coordinate p to
 * a flat offset at which that element would be stored.
 *
 * If the matrix is being accessed in transposed form, the
 * orient flag effectively swaps p=(i,j) => (j,i) behind
 * the scenes.
 *
 * Returns -1 if there is no matrix element associated with p.
 */
typedef base_int_t (*mpi_matrix_coord_index_to_offset_callback)(mpi_matrix_coord_ptr coord, mpi_matrix_orient_t orient, int_pair_t p);

/*
 * @typedef mpi_matrix_coord_callbacks_t
 *
 * The list of callback functions associated with a matrix
 * coordinate instance.
 */
typedef struct {
    mpi_matrix_coord_element_count_callback     element_count;
    mpi_matrix_coord_index_status_callback      index_status;
    mpi_matrix_coord_index_reduce_callback      index_reduce;
    mpi_matrix_coord_index_to_offset_callback   index_to_offset;
} mpi_matrix_coord_callbacks_t;

/*
 * @typedef mpi_matrix_coord_t
 *
 * The data structure of a matrix coordinate system.  All fields
 * are filled-in by the mpi_matrix_coord_init() function -- most
 * importantly, the callbacks associated with the type.
 *
 * There is no dynamically-allocated or additional storage associated
 * with the types, so a mpi_matrix_coord_t can be used in other data
 * structures and as a local variable on the stack.
 *
 * While it is permissible for external code to fill-in alternative
 * callbacks and dimensions and also use this API, external code
 * should not alter the fields once the mpi_matrix_coord_t has been
 * initialized.
 */
typedef struct mpi_matrix_coord {
    mpi_matrix_coord_type_t         type;
    bool                            is_row_major;
    int_pair_t                      dimensions;
    base_int_t                      k1, k2; /* optional additional integer params */
    mpi_matrix_coord_callbacks_t    callbacks;
} mpi_matrix_coord_t;

/*
 * @function mpi_matrix_coord_create
 *
 * Dynamically allocate a new matrix coordinate system of the given
 * type and dimensions.  The mpi_matrix_coord_init() function is
 * called to initialize the new instance so it does not need to be
 * called a second time.
 *
 * Returns NULL if the instance could not be allocated.
 */
mpi_matrix_coord_ptr mpi_matrix_coord_create(mpi_matrix_coord_type_t type, bool is_row_major, int_pair_t dimensions, ...);

/*
 * @function mpi_matrix_coord_init
 *
 * Used to initialize a static instance of the data structure, e.g.
 * a local variable on the stack or a member of another data
 * structure.
 *
 * Fills-in the callbacks fields with the appropriate functions for
 * the given type.  Otherwise, the entire data structure could just
 * as easily be configured via an initialization list.
 *
 * Returns coord.
 */
mpi_matrix_coord_ptr mpi_matrix_coord_init(mpi_matrix_coord_ptr coord, mpi_matrix_coord_type_t type, bool is_row_major, int_pair_t dimensions, ...);

/*
 * @function mpi_matrix_coord_destroy
 *
 * Destroy a dynamically-allocated instance of mpi_matrix_coord.
 */
void mpi_matrix_coord_destroy(mpi_matrix_coord_ptr coord);

/*
 * @defined mpi_matrix_coord_get_type_name
 *
 * Convenience wrapper to getting the textual description of
 * the instance's matrix coordinate system type.
 */
#define mpi_matrix_coord_get_type_name(C) (mpi_matrix_coord_type_get_name((C)->type))

/*
 * @defined mpi_matrix_coord_element_count
 *
 * Convenience wrapper to the instance's element_count callback.
 */
#define mpi_matrix_coord_element_count(C) ((C)->callbacks.element_count((C)))

/*
 * @defined mpi_matrix_coord_index_status
 *
 * Convenience wrapper to the instance's index_status callback.
 */
#define mpi_matrix_coord_index_status(C, T, P) ((C)->callbacks.index_status((C), (T), (P)))

/*
 * @defined mpi_matrix_coord_index_reduce
 *
 * Convenience wrapper to the instance's index_reduce callback.
 */
#define mpi_matrix_coord_index_reduce(C, T, P) ((C)->callbacks.index_reduce((C), (T), (P)))

/*
 * @defined mpi_matrix_coord_index_to_offset
 *
 * Convenience wrapper to the instance's index_to_offset callback.
 */
#define mpi_matrix_coord_index_to_offset(C, T, P) ((C)->callbacks.index_to_offset((C), (T), (P)))

#endif /* __MPI_MATRIX_COORD_H__ */
