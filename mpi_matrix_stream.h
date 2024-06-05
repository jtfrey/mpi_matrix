/*	mpi_matrix_stream.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header MPI matrix element streaming
	
	A pseudo-class that handles the aggregation of matrix elements for
	communication to other ranks.
	
	A frame is a unit of implicit or explicit matrix elements.  An implicit
	frame has an implied singular value and an element count.  An explicit
	frame has an element count and ordered list of values.  An ordered set
	of implicit and explicit frames constitutes a frame buffer.
	
	Streaming a single column of a dense matrix would likely be a matter of
	the server and client agreeing upon a frame buffer byte_capacity; the
	server fills its frame buffer with a single explicit frame, sends that
	frame to the client; the client receives the frame and processes the
	values from that explicit frame.  While the client is processing its copy
	of the frame buffer, the server is refilling its buffer with the next
	chunk of values to repeat the loop until it has finished sending the
	entire column.
	
	The main benefit of implicit frames lies in optimizing the client
	processing.  Consider a client implementing a dot product with an implied
	value of zero.  When the client encounters an implicit frame of 100
	elements, it skips those 100 elements (and the 100 elements of the
	second vector) without doing any computation:  each element will just add
	zero to the sum.
	
	It is possible to mix implicit and explicit frames with a dense matrix
	if the consumer code is written such that it filters the values as they
	are added to the frame buffer.  A range of the implied value, when
	recognized, yields an implicit frame.  A small dense matrix with many
	explicit zeroes may not be an optimal candidate for representation as
	sparse, but streaming its matrix elements would likely benefit from
	eliminating zeroes as implicit frames.
	
	Naturally, a sparse matrix is best-streamed as a mix of implicit and
	explicit frames with implied value zero.
	
	On the client side, the frame buffer has data written into it by a
	call to MPI_Recv(), for example.  Iterating over the frame buffer then
	looks like:
	
	    mpi_matrix_framebuffer_ref  fb;
	    double                      dot = 0.0, *our_values;
	    
	    MPI_Recv(
	        mpi_matrix_framebuffer_get_buffer_ptr(fb),
	        mpi_matrix_framebuffer_get_byte_capacity(fb),
	        MPI_BYTE,
	        …);
	    if ( mpi_matrix_framebuffer_is_valid(fb) ) {
	        const void  *iter = NULL;
	        
	        while ( (iter = mpi_matrix_framebuffer_iter_next_frame(fb, iter)) ) {
	            base_int_t  n_elements = mpi_matrix_framebuffer_iter_get_n_elements(fb, iter);
	            
	            if ( n_elements > 0 ) {
	                double      *their_values;
	                
	                if ( ! mpi_matrix_framebuffer_iter_get_elements_ptr(fb, iter, &their_values) ) break;
	                while ( n_elements-- ) dot += *our_values++ * *their_values++;
	            } else {
	                our_values += -n_elements;
	            }
	        }
	    }
	        
*/

#ifndef __MPI_MATRIX_STREAM_H__
#define __MPI_MATRIX_STREAM_H__

#include "mpi_matrix_config.h"
#include "mpi_matrix_storage.h"

/*
 * @typedef mpi_matrix_framebuffer_ref
 *
 * Opaque reference to an MPI matrix streaming frame buffer.
 */
typedef struct mpi_matrix_framebuffer * mpi_matrix_framebuffer_ref;

/*
 * @function mpi_matrix_framebuffer_create
 *
 * Allocate and initialize a new frame buffer that will compile
 * matrix elements of the given datatype.  The new frame buffer
 * will accept at most byte_capacity bytes-worth of frames.
 *
 * Returns NULL if the frame buffer could not be allocated.
 */
mpi_matrix_framebuffer_ref mpi_matrix_framebuffer_create(mpi_matrix_storage_datatype_t datatype, base_int_t byte_capacity);

/*
 * @function mpi_matrix_framebuffer_destroy
 *
 * Deallocate the frame buffer fb.
 */
void mpi_matrix_framebuffer_destroy(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_get_datatype
 *
 * Returns the underlying numerical data type associated with
 * fb.
 */
mpi_matrix_storage_datatype_t mpi_matrix_framebuffer_get_datatype(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_get_byte_capacity
 *
 * Returns the byte capacity of the frame buffer associated with
 * fb.
 */
size_t mpi_matrix_framebuffer_get_byte_capacity(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_get_bytes_unused
 *
 * Returns the unused byte capacity of the frame buffer associated with
 * fb.
 */
size_t mpi_matrix_framebuffer_get_bytes_unused(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_get_buffer_ptr
 *
 * Returns the base pointer of the compiled frame buffer associated
 * with fb.
 */
const void* mpi_matrix_framebuffer_get_buffer_ptr(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_reset
 *
 * Discard all frames currently compiled in fb.
 */
void mpi_matrix_framebuffer_reset(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_push_explicit
 *
 * Given an array of n_elements at elements, attempt to compile an
 * explicit frame into fb.  On return, *n_elements_remaining will be
 * zero if all n_elements were compiled into the frame or a value
 * 0 < *n_elements_remaining <= n_elements if some subset of elements
 * were added (n_elements - *n_elements_remaining).
 *
 * Returns true if at least one element was compiled into fb, false
 * if nothing could be added.  Thus, the conditions for a server to
 * send its frame buffer to the client after calling this function
 * are a return value of false or true with non-zero
 * *n_elements_remaining.
 */
bool mpi_matrix_framebuffer_push_explicit(mpi_matrix_framebuffer_ref fb, base_int_t n_elements, void *elements, base_int_t *n_elements_remaining);

/*
 * @function mpi_matrix_framebuffer_push_implicit
 *
 * Attempt to compile an implicit frame of n_elements into fb.  On
 * return, *n_elements_remaining will be zero if successful and
 * n_elements if not.
 *
 * Returns true if the elements were compiled into fb, false
 * if not.  Thus, the conditions for a server to send its frame buffer
 * to the client after calling this function is a return value of
 * false.
 */
bool mpi_matrix_framebuffer_push_implicit(mpi_matrix_framebuffer_ref fb, base_int_t n_elements, base_int_t *n_elements_remaining);

#ifdef HAVE_STD_C11
#   ifdef HAVE_STD_C23
/*
 * @defined mpi_matrix_framebuffer_push
 *
 * Generic select of explicit vs. implicit framebuffer push based
 * on the type of the third argument.  If the argument is a void pointer
 * or pointer to any of the four floating point types available to the
 * API, the call is assumed to be an explicit push.  If the third
 * argument is a base_int_t pointer, the call is assumed to be an
 * implicit push.
 */
#       define mpi_matrix_framebuffer_push(FB, NE, E, ...) _Generic((E), \
                    void*: mpi_matrix_framebuffer_push_explicit, \
                    float*: mpi_matrix_framebuffer_push_explicit, \
                    double*: mpi_matrix_framebuffer_push_explicit, \
                    complex float*: mpi_matrix_framebuffer_push_explicit, \
                    complex double*: mpi_matrix_framebuffer_push_explicit, \
                    base_int_t*: mpi_matrix_framebuffer_push_implicit \
                )((FB), (NE), (E) __VA_OPT__(,) __VA_ARGS__)
#   else
#       define mpi_matrix_framebuffer_push(FB, NE, E, ...) _Generic((E), \
                    void*: mpi_matrix_framebuffer_push_explicit, \
                    float*: mpi_matrix_framebuffer_push_explicit, \
                    double*: mpi_matrix_framebuffer_push_explicit, \
                    complex float*: mpi_matrix_framebuffer_push_explicit, \
                    complex double*: mpi_matrix_framebuffer_push_explicit, \
                    base_int_t*: mpi_matrix_framebuffer_push_implicit \
                )((FB), (NE), (E), ##__VA_ARGS__)
#   endif
#endif

/*
 * @function mpi_matrix_framebuffer_is_valid
 *
 * Verify that the compiled frame buffer present in fb is valid.  Intended
 * to be used after receiving a frame buffer into fb via MPI, for example.
 * Any time data is imported into fb from an external source, this function
 * MUST be called before doing anything else with fb.
 */
bool mpi_matrix_framebuffer_is_valid(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_get_frame_count
 *
 * Returns the number of frames present in the frame buffer associated
 * with fb.
 */
base_int_t mpi_matrix_framebuffer_get_frame_count(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_get_element_count
 *
 * Returns the number of elements present across all frames in the frame buffer
 * associated with fb.
 */
base_int_t mpi_matrix_framebuffer_get_element_count(mpi_matrix_framebuffer_ref fb);

/*
 * @function mpi_matrix_framebuffer_iter_next_frame
 *
 * Iterate over the frames present in the frame buffer associated with
 * fb.  The iteration must start with a call for which iter_context is
 * NULL, and continues passing the last-returned pointer until the
 * function returns NULL:
 *
 *     const void  *iter_context = NULL;
 *
 *     while ( (iter_context = mpi_matrix_framebuffer_iter_next_frame(fb, iter_context)) ) {
 *         // Do something with current frame...
 *     }
 */
const void* mpi_matrix_framebuffer_iter_next_frame(mpi_matrix_framebuffer_ref fb, const void *iter_context);

/*
 * @function mpi_matrix_framebuffer_iter_get_n_elements
 *
 * After a call to mpi_matrix_framebuffer_iter_next_frame that returns
 * a non-NULL pointer, this function retrieves the element count associated
 * with the frame.
 *
 * A negative non-zero return value indicates an implicit frame consisting
 * of an element count equal to the absolute value, e.g. -24 => 24 elements
 * with the implied value.
 *
 * A positive non-zero return value indicates an explicit frame consisting
 * of that many values.
 *
 * A zero return value indicates an error (e.g. iteration has completed).
 */
base_int_t mpi_matrix_framebuffer_iter_get_n_elements(mpi_matrix_framebuffer_ref fb, const void *iter_context);
    
/*
 * @function mpi_matrix_framebuffer_iter_get_elements_ptr
 *
 * After a call to mpi_matrix_framebuffer_iter_next_frame that returns
 * a non-NULL pointer, this function sets *elements_ptr to the base of the
 * array of explicit values.
 *
 * Returns true if *elements_ptr is set, false otherwise (e.g. iteration has
 * completed).
 */
bool mpi_matrix_framebuffer_iter_get_elements_ptr(mpi_matrix_framebuffer_ref fb, const void *iter_context, void* *elements_ptr);

#endif /* __MPI_MATRIX_STREAM_H__ */
