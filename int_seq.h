/*	int_seq.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header Integer sequence
	
	Representation and functionality for a range of
	integers with an arbitrary step size.  The sequence is
	implemented as a range and integer step size.  Note that
	negative length and step size are permissible.
	
	All functions are implemented as static inline in the hopes
	that the compiler will optimize-out actual function calls.
*/

#ifndef __INT_SEQ_H__
#define __INT_SEQ_H__

#include "mpi_matrix_config.h"

/*
 * @typedef int_seq_t
 *
 * An integer sequence running from start to (start + length ± 1).
 */
typedef struct {
    base_int_t  start, length, step;
} int_seq_t;

/*
 * @function int_seq_make
 *
 * Initialize and return a new int_seq_t with the given
 * start, length, and step size.
 *
 * The step size should have the same sign as the length
 * for the sequence to work properly.
 */
static inline int_seq_t
int_seq_make(
    base_int_t  start,
    base_int_t  length,
    base_int_t  step
)
{
    int_seq_t s = { .start = start, .length = length, .step = step };
    return s;
}

/*
 * @function int_seq_make_with_start_and_end
 *
 * Initialize and return a new int_seq_t starting at the
 * start value and with a length that ends the sequence at the
 * end value.  
 *
 * The step size should have the same sign as the (end - start)
 * for the sequence to work properly.
 */
static inline int_seq_t
int_seq_make_with_start_and_end(
    base_int_t  start,
    base_int_t  end,
    base_int_t  step
)
{
    int_seq_t   s = {   .start = start,
                        .length = (end - start) + ((end < start) ? -1 : +1),
                        .step = step };
    return s;
}

/*
 * @function int_seq_is_equal
 *
 * Returns true if the sequences s1 and s2 cover the exact
 * same integer values.
 */
static inline bool
int_seq_is_equal(
    int_seq_t   s1,
    int_seq_t   s2
)
{
    return ((s1.start == s2.start) && (s1.length == s2.length) && (s1.step == s2.step));
}

/*
 * @function int_seq_is_valid
 *
 * Returns true if the sequence s has a non-zero step with the
 * same sign as length.
 */
static inline bool
int_seq_is_valid(
    int_seq_t   s
)
{
    if ( s.step < 0 ) {
        return ((s.length >= 0) || (INT_MIN - s.length > s.start));
    } else {
        return ((s.length <= 0) || (INT_MAX - s.length < s.start));
    }
}

/*
 * @function int_seq_get_end
 *
 * Returns the final integer value in the sequence s.
 */
static inline base_int_t
int_seq_get_end(
    int_seq_t   s
)
{
    return s.start + s.length + ((s.step < 0) ? +1 : -1);
}

/*
 * @function int_seq_get_max
 *
 * Returns the first integer value beyond the sequence s.
 */
static inline base_int_t
int_seq_get_max(
    int_seq_t   s
)
{
    return s.start + s.length;
}

/*
 * @function int_seq_get_count
 *
 * Returns the number of integer values visited by sequence s.
 */
static inline base_int_t
int_seq_get_count(
    int_seq_t   s
)
{
    return (s.length / s.step);
}

/*
 * @function int_seq_get_last
 *
 * Returns the last integer value visited by sequence s.
 */
static inline base_int_t
int_seq_get_last(
    int_seq_t   s
)
{
    base_int_t  d = (s.length / s.step);
    return s.start + s.step * d;
}

/*
 * @function int_seq_is_reverse
 *
 * Returns true if the sequences s1 and s2 cover the exact
 * same integer values in opposite order.
 */
static inline bool
int_seq_is_reverse(
    int_seq_t   s1,
    int_seq_t   s2
)
{
    return ((s1.start == int_seq_get_end(s2)) && (s1.length == -s2.length) && (s1.step == -s2.step));
}

/*
 * @function int_seq_does_contain
 *
 * Returns true if the integer i is visited within the given
 * sequence s.
 */
static inline bool
int_seq_does_contain(
    int_seq_t   s,
    base_int_t  i
)
{
    if ( s.step < 0 ) {
        return ((i > int_seq_get_max(s)) && (i <= s.start) && (((i - s.start) % -s.step) == 0));
    } else {
        return ((i >= s.start) && (i < int_seq_get_max(s)) && (((i - s.start) % s.step) == 0));
    }
}

/*
 * @defined int_seq_iter_start
 *
 * Given the sequence S, generate a looping construct that iterates
 * over every integer in the sequence from start to end.  The value at
 * each iteration will be assigned to the variable named according
 * to I_VAR.
 *
 * The int_seq_iter_start() macro must be balanced by a matching
 * int_seq_iter_end() macro.
 *
 * E.g.
 *
 *     int_seq_iter_start(S, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_seq_iter_end(S, idx)
 */
#define int_seq_iter_start(S, I_VAR) \
    { \
        base_int_t  I_VAR = (S).start, \
                    S ## _count = int_seq_get_count((S)); \
        while ( S ## _count-- > 0 ) {

/*
 * @defined int_seq_iter_end
 *
 * An int_seq_iter_start() macro must be balanced by a matching
 * int_seq_iter_end() macro.
 *
 * E.g.
 *
 *     int_seq_iter_start(S, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_seq_iter_end(S, idx)
 */
#define int_seq_iter_end(S, I_VAR) \
            I_VAR += (S).step; \
        } \
    }

/*
 * @defined int_seq_reverse_iter_start
 *
 * Given the sequence S, generate a looping construct that iterates
 * over every integer in the sequence from end to start.  The value at
 * each iteration will be assigned to the variable named according
 * to I_VAR.
 *
 * The int_seq_reverse_iter_start() macro must be balanced by a matching
 * int_seq_reverse_iter_end() macro.
 *
 * E.g.
 *
 *     int_seq_reverse_iter_start(S, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_seq_reverse_iter_end(S, idx)
 */
#define int_seq_reverse_iter_start(S, I_VAR) \
    { \
        base_int_t  I_VAR = int_seq_get_last((S)), \
                    S ## _count = int_seq_get_count((S)); \
        while ( S ## _count-- > 0 ) { \

/*
 * @defined int_seq_reverse_iter_end
 *
 * An int_seq_reverse_iter_start() macro must be balanced by a matching
 * int_seq_reverse_iter_end() macro.
 *
 * E.g.
 *
 *     int_seq_reverse_iter_start(S, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_seq_reverse_iter_end(S, idx)
 */
#define int_seq_reverse_iter_end(S, I_VAR) \
            I_VAR -= (S).step; \
        } \
    }

#endif /* __INT_SEQ_H__ */
