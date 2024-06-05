/*	int_range.h
	Copyright (c) 2024, J T Frey
*/

/*!
	@header Integer range
	
	Representation and functionality for a range of
	integers.  The range is implemented as the pair (start, length)
	rather than (start,end), though functions are present to
	calculate end from (start, length).
	
	All functions are implemented as static inline in the hopes
	that the compiler will optimize-out actual function calls.
*/

#ifndef __INT_RANGE_H__
#define __INT_RANGE_H__

#include "mpi_matrix_config.h"

/*
 * @typedef int_range_t
 *
 * An integer range running from start to (start + length - 1).
 */
typedef struct {
    base_int_t  start, length;
} int_range_t;

/*
 * @function int_range_make
 *
 * Initialize and return a new int_range_t with the given
 * start and length.
 */
static inline int_range_t
int_range_make(
    base_int_t  start,
    base_int_t  length
)
{
    int_range_t r = { .start = start, .length = length };
    return r;
}

/*
 * @function int_range_make_with_low_and_high
 *
 * Initialize and return a new int_range_t starting at the
 * low value and with a length that ends the range at the
 * high value, e.g. [low,high].
 */
static inline int_range_t
int_range_make_with_low_and_high(
    base_int_t  low,
    base_int_t  high
)
{
    int_range_t r = { .start = low, .length = (high - low) + 1 };
    return r;
}

/*
 * @function int_range_is_equal
 *
 * Returns true if the ranges r1 and r2 cover the exact
 * same integer values.
 */
static inline bool
int_range_is_equal(
    int_range_t r1,
    int_range_t r2
)
{
    return ((r1.start == r2.start) && (r1.length == r2.length));
}

/*
 * @function int_range_is_valid
 *
 * Returns true if the range r has non-negative length and does
 * not extend beyond the range of an int.
 */
static inline bool
int_range_is_valid(
    int_range_t r
)
{
    return ((r.length <= 0) || (INT_MAX - r.length < r.start));
}

/*
 * @function int_range_get_end
 *
 * Returns the final integer value in the range r.
 */
static inline base_int_t
int_range_get_end(
    int_range_t r
)
{
    return r.start + r.length - 1;
}

/*
 * @function int_range_get_max
 *
 * Returns the first integer value beyond the range r.
 */
static inline base_int_t
int_range_get_max(
    int_range_t r
)
{
    return r.start + r.length;
}

/*
 * @function int_range_does_contain
 *
 * Returns true if the integer i lies within the given
 * range r.
 */
static inline bool
int_range_does_contain(
    int_range_t r,
    base_int_t  i
)
{
    return ((i >= r.start) && ((i - r.start) < r.length));
}

/*
 * @function int_range_is_adjacent
 *
 * Returns true if the two ranges do not intersect but together
 * span a single range of integer values.
 */
static inline bool
int_range_is_adjacent(
    int_range_t r1,
    int_range_t r2
)
{
    return ((r1.start + r1.length == r2.start) || (r2.start + r2.length == r1.start));
}

/*
 * @function int_range_does_intersect
 *
 * Returns true if the two ranges overlap (include at least one
 * integer value in common).
 */
static inline bool
int_range_does_intersect(
    int_range_t r1,
    int_range_t r2
)
{
    base_int_t  end1 = r1.start + r1.length - 1;
    base_int_t  end2 = r2.start + r2.length - 1;
    
    return (r1.start < r2.start) ? (end1 >= r2.start) : (end2 >= r1.start);
}

/*
 * @function int_range_is_adjacent_or_intersecting
 *
 * Returns true if the two ranges either intersect or lie
 * adjacent to each other.
 */
static inline bool
int_range_is_adjacent_or_intersecting(
    int_range_t r1,
    int_range_t r2
)
{
    if (r1.start + r1.length == r2.start) return true;
    if (r2.start + r2.length == r1.start) return true;
    
    base_int_t  end1 = r1.start + r1.length - 1;
    base_int_t  end2 = r2.start + r2.length - 1;
    
    return (r1.start < r2.start) ? (end1 >= r2.start) : (end2 >= r1.start);
}

/*
 * @function int_range_intersection
 *
 * If the two ranges intersect, the returned range contains
 * the integers r1 and r2 have in common.
 *
 * If the two ranges do not intersect, the result will be
 * invalid (negative length).
 */
static inline int_range_t
int_range_intersection(
    int_range_t r1,
    int_range_t r2
)
{
    int_range_t r;
    base_int_t  end1 = r1.start + r1.length - 1;
    base_int_t  end2 = r2.start + r2.length - 1;
    
    r.start = (r1.start < r2.start) ? r2.start : r1.start;
    r.length = ((end1 < end2) ? end1 : end2) - r.start + 1;
    return r;
}

/*
 * @function int_range_union
 *
 * If the two ranges intersect, the returned range contains
 * the integers spanned by r1 and r2 in aggregate.
 *
 * If the two ranges do not intersect, the result will be
 * invalid.
 */
static inline int_range_t
int_range_union(
    int_range_t r1,
    int_range_t r2
)
{
    int_range_t r;
    base_int_t  end1 = r1.start + r1.length - 1;
    base_int_t  end2 = r2.start + r2.length - 1;
    
    r.start = (r1.start < r2.start) ? r1.start : r2.start;
    r.length = ((end1 < end2) ? end2 : end1) - r.start + 1;
    return r;
}

/*
 * @defined int_range_iter_start
 *
 * Given the range R, generate a looping construct that iterates
 * over every integer in the range from start to end.  The value at
 * each iteration will be assigned to the variable named according
 * to I_VAR.
 *
 * The int_range_iter_start() macro must be balanced by a matching
 * int_range_iter_end() macro.
 *
 * E.g.
 *
 *     int_range_iter_start(R, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_range_iter_end(R, idx)
 */
#define int_range_iter_start(R, I_VAR) \
    { \
        base_int_t  I_VAR = (R).start, \
                    R ## _count = (R).length; \
        while ( R ## _count-- ) {

/*
 * @defined int_range_iter_end
 *
 * An int_range_iter_start() macro must be balanced by a matching
 * int_range_iter_end() macro.
 *
 * E.g.
 *
 *     int_range_iter_start(R, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_range_iter_end(R, idx)
 */
#define int_range_iter_end(R, I_VAR) \
            I_VAR++; \
        } \
    }

/*
 * @defined int_range_reverse_iter_start
 *
 * Given the range R, generate a looping construct that iterates
 * over every integer in the range from end to start.  The value at
 * each iteration will be assigned to the variable named according
 * to I_VAR.
 *
 * The int_range_reverse_iter_start() macro must be balanced by a matching
 * int_range_reverse_iter_end() macro.
 *
 * E.g.
 *
 *     int_range_reverse_iter_start(R, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_range_reverse_iter_end(R, idx)
 */
#define int_range_reverse_iter_start(R, I_VAR) \
    { \
        base_int_t  I_VAR = int_range_get_max((R)), \
        while ( I_VAR-- > (R).start ) {

/*
 * @defined int_range_reverse_iter_end
 *
 * The int_range_reverse_iter_start() macro must be balanced by a matching
 * int_range_reverse_iter_end() macro.
 *
 * E.g.
 *
 *     int_range_reverse_iter_start(R, idx)
 *         // This is the loop body...
 *         printf("%d\n", (int)idx);
 *     int_range_reverse_iter_end(R, idx)
 */
#define int_range_reverse_iter_end(R, I_VAR) \
        } \
    }

#endif /* __INT_RANGE_H__ */
