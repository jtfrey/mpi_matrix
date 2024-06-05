
#include "mpi_matrix_stream.h"

//

static inline base_int_t
__base_int_abs(
    base_int_t  i
)
{
    return (i < 0) ? -i : i;
}

//

typedef struct {
    base_int_t      n_elements;
} mpi_matrix_frame_implicit_t;

typedef struct {
    base_int_t      n_elements;
    float           elements[0];
} mpi_matrix_frame_explicit_real_sp_t;

typedef struct {
    base_int_t      n_elements;
    double          elements[0];
} mpi_matrix_frame_explicit_real_dp_t;

typedef struct {
    base_int_t      n_elements;
    float complex   elements[0];
} mpi_matrix_frame_explicit_complex_sp_t;

typedef struct {
    base_int_t      n_elements;
    double complex  elements[0];
} mpi_matrix_frame_explicit_complex_dp_t;

//

static size_t __mpi_matrix_frame_byte_sizes[] = {
                sizeof(mpi_matrix_frame_explicit_real_sp_t),
                sizeof(mpi_matrix_frame_explicit_real_dp_t),
                sizeof(mpi_matrix_frame_explicit_complex_sp_t),
                sizeof(mpi_matrix_frame_explicit_complex_dp_t),
                0
            };

//

static inline size_t
__mpi_matrix_frame_base_byte_size(
    mpi_matrix_storage_datatype_t   datatype,
    base_int_t                      n_elements
)
{
    return (n_elements < 0) ? sizeof(mpi_matrix_frame_implicit_t) :
                __mpi_matrix_frame_byte_sizes[datatype] + n_elements * mpi_matrix_storage_datatype_byte_sizes[datatype];
}

static inline size_t
__mpi_matrix_frame_delta_byte_size(
    mpi_matrix_storage_datatype_t   datatype,
    base_int_t                      n_elements
)
{
    return (n_elements <= 0) ? 0 : n_elements * mpi_matrix_storage_datatype_byte_sizes[datatype];
}

static inline base_int_t
__mpi_matrix_frame_max_elements_allocable(
    mpi_matrix_storage_datatype_t   datatype,
    size_t                          unallocated_bytes
)
{
    return (unallocated_bytes >= __mpi_matrix_frame_byte_sizes[datatype]) ? (unallocated_bytes - __mpi_matrix_frame_byte_sizes[datatype]) / mpi_matrix_storage_datatype_byte_sizes[datatype] : 0;
}

static inline void*
__mpi_matrix_frame_explicit_base_elements_ptr(
    mpi_matrix_storage_datatype_t   datatype,
    void                            *frame_ptr
)
{
    switch ( datatype ) {
        case mpi_matrix_storage_datatype_real_sp:
            return (void*)&((mpi_matrix_frame_explicit_real_sp_t*)frame_ptr)->elements[0];
        case mpi_matrix_storage_datatype_real_dp:
            return (void*)&((mpi_matrix_frame_explicit_real_dp_t*)frame_ptr)->elements[0];
        case mpi_matrix_storage_datatype_complex_sp:
            return (void*)&((mpi_matrix_frame_explicit_complex_sp_t*)frame_ptr)->elements[0];
        case mpi_matrix_storage_datatype_complex_dp:
            return (void*)&((mpi_matrix_frame_explicit_complex_dp_t*)frame_ptr)->elements[0];
        case mpi_matrix_storage_datatype_max:
            break;
    }
    return NULL;
}

static inline void*
__mpi_matrix_frame_explicit_end_elements_ptr(
    mpi_matrix_storage_datatype_t   datatype,
    void                            *frame_ptr
)
{
    switch ( datatype ) {
        case mpi_matrix_storage_datatype_real_sp:
            return (void*)&((mpi_matrix_frame_explicit_real_sp_t*)frame_ptr)->elements[((mpi_matrix_frame_explicit_real_sp_t*)frame_ptr)->n_elements];
        case mpi_matrix_storage_datatype_real_dp:
            return (void*)&((mpi_matrix_frame_explicit_real_dp_t*)frame_ptr)->elements[((mpi_matrix_frame_explicit_real_dp_t*)frame_ptr)->n_elements];
        case mpi_matrix_storage_datatype_complex_sp:
            return (void*)&((mpi_matrix_frame_explicit_complex_sp_t*)frame_ptr)->elements[((mpi_matrix_frame_explicit_complex_sp_t*)frame_ptr)->n_elements];
        case mpi_matrix_storage_datatype_complex_dp:
            return (void*)&((mpi_matrix_frame_explicit_complex_dp_t*)frame_ptr)->elements[((mpi_matrix_frame_explicit_complex_dp_t*)frame_ptr)->n_elements];
        case mpi_matrix_storage_datatype_max:
            break;
    }
    return NULL;
}

static inline void
__mpi_matrix_frame_explicit_add_element_count(
    mpi_matrix_storage_datatype_t   datatype,
    void                            *frame_ptr,
    base_int_t                      n_elements
)
{
    switch ( datatype ) {
        case mpi_matrix_storage_datatype_real_sp:
            ((mpi_matrix_frame_explicit_real_sp_t*)frame_ptr)->n_elements += n_elements;
            break;
        case mpi_matrix_storage_datatype_real_dp:
            ((mpi_matrix_frame_explicit_real_dp_t*)frame_ptr)->n_elements += n_elements;
            break;
        case mpi_matrix_storage_datatype_complex_sp:
            ((mpi_matrix_frame_explicit_complex_sp_t*)frame_ptr)->n_elements += n_elements;
            break;
        case mpi_matrix_storage_datatype_complex_dp:
            ((mpi_matrix_frame_explicit_complex_dp_t*)frame_ptr)->n_elements += n_elements;
            break;
        case mpi_matrix_storage_datatype_max:
            break;
    }
}

static inline base_int_t
__mpi_matrix_frame_explicit_get_element_count(
    mpi_matrix_storage_datatype_t   datatype,
    void                            *frame_ptr
)
{
    switch ( datatype ) {
        case mpi_matrix_storage_datatype_real_sp:
            return ((mpi_matrix_frame_explicit_real_sp_t*)frame_ptr)->n_elements;
        case mpi_matrix_storage_datatype_real_dp:
            return ((mpi_matrix_frame_explicit_real_dp_t*)frame_ptr)->n_elements;
        case mpi_matrix_storage_datatype_complex_sp:
            return ((mpi_matrix_frame_explicit_complex_sp_t*)frame_ptr)->n_elements;
        case mpi_matrix_storage_datatype_complex_dp:
            return ((mpi_matrix_frame_explicit_complex_dp_t*)frame_ptr)->n_elements;
        case mpi_matrix_storage_datatype_max:
            break;
    }
    return 0;
}

static inline void
__mpi_matrix_frame_implicit_add_element_count(
    mpi_matrix_storage_datatype_t   datatype,
    void                            *frame_ptr,
    base_int_t                      n_elements
)
{
    ((mpi_matrix_frame_implicit_t*)frame_ptr)->n_elements -= n_elements;
}

static inline base_int_t
__mpi_matrix_frame_implicit_get_element_count(
    mpi_matrix_storage_datatype_t   datatype,
    void                            *frame_ptr
)
{
    return ((mpi_matrix_frame_implicit_t*)frame_ptr)->n_elements;
}

//

typedef struct {
    base_int_t      n_frames;
    unsigned char   frames[0];
} mpi_matrix_frames_t;

typedef struct mpi_matrix_framebuffer {
    mpi_matrix_storage_datatype_t   datatype;
    base_int_t                      byte_capacity;
    mpi_matrix_frames_t             *frames_base;
    void                            *frames_current, *frames_next, *frames_end;
} mpi_matrix_framebuffer_t;

//

static inline size_t
__mpi_matrix_framebuffer_frames_byte_size(
    mpi_matrix_framebuffer_t    *fb
)
{
    return fb->frames_next - (void*)fb->frames_base;
}

//

bool
__mpi_matrix_framebuffer_alloc_frame(
    mpi_matrix_framebuffer_t    *fb,
    base_int_t                  *n_elements
)
{
    if ( *n_elements > 0 ) {
        base_int_t          max_elements = __mpi_matrix_frame_max_elements_allocable(fb->datatype, fb->frames_end - fb->frames_next);
        
        if ( max_elements > 0 ) {
            size_t         frame_size;
            
            // Fall back to the actual number of elements if the frame buffer has
            // more than sufficient space:
            if ( *n_elements < max_elements ) max_elements = *n_elements;
            
            // Determine how large the new frame will be:
            frame_size = __mpi_matrix_frame_base_byte_size(fb->datatype, max_elements);
        
            // Allocate the new frame:
            fb->frames_current = fb->frames_next;
            fb->frames_next += frame_size;
            fb->frames_base->n_frames++;
            
            // Initialize the new frame; the calling code will take care of copying
            // the elements into the buffer:
            memset(fb->frames_current, 0, frame_size);
            __mpi_matrix_frame_explicit_add_element_count(fb->datatype, fb->frames_current, max_elements);
            *n_elements -= max_elements;
            return true;
        }
    }
    else if ( *n_elements < 0 ) {
        if ( sizeof(mpi_matrix_frame_implicit_t) <= (fb->frames_end - fb->frames_next) ) {
            // Allocate the new frame:
            fb->frames_current = fb->frames_next;
            fb->frames_next += sizeof(mpi_matrix_frame_implicit_t);
            fb->frames_base->n_frames++;
            
            // Initialize the new frame:
            ((mpi_matrix_frame_implicit_t*)fb->frames_current)->n_elements = *n_elements;
            *n_elements = 0;
            
            return true;
        }
    }
    return false;
}

//

mpi_matrix_framebuffer_ref
mpi_matrix_framebuffer_create(
    mpi_matrix_storage_datatype_t   datatype,
    base_int_t                      byte_capacity
)
{
    size_t                      byte_size = sizeof(mpi_matrix_framebuffer_t) +
                                                    byte_capacity;
    mpi_matrix_framebuffer_t    *new_fb = (mpi_matrix_framebuffer_t*)malloc(byte_size);
    
    if ( new_fb ) {
        new_fb->datatype = datatype;
        new_fb->byte_capacity = byte_capacity;
        
        new_fb->frames_base = (mpi_matrix_frames_t*)((void*)new_fb + sizeof(mpi_matrix_framebuffer_t));
        new_fb->frames_base->n_frames = 0;
        
        new_fb->frames_current = NULL;
        new_fb->frames_next = (void*)new_fb->frames_base + sizeof(mpi_matrix_frames_t);
        new_fb->frames_end = (void*)new_fb->frames_base + byte_capacity;
    }
    return new_fb;
}

//

void
mpi_matrix_framebuffer_destroy(
    mpi_matrix_framebuffer_ref  fb
)
{
    free((void*)fb);
}

//

mpi_matrix_storage_datatype_t
mpi_matrix_framebuffer_get_datatype(
    mpi_matrix_framebuffer_ref  fb
)
{
    return fb->datatype;
}

//

size_t
mpi_matrix_framebuffer_get_byte_capacity(
    mpi_matrix_framebuffer_ref  fb
)
{
    return fb->byte_capacity;
}

//

size_t
mpi_matrix_framebuffer_get_bytes_unused(
    mpi_matrix_framebuffer_ref  fb
)
{
    return fb->frames_end - fb->frames_next;
}

//

const void*
mpi_matrix_framebuffer_get_buffer_ptr(
    mpi_matrix_framebuffer_ref  fb
)
{
    return (const void*)fb->frames_base;
}

//

void
mpi_matrix_framebuffer_reset(
    mpi_matrix_framebuffer_ref  fb
)
{
    fb->frames_base->n_frames = 0;
    fb->frames_current = NULL;
    fb->frames_next = (void*)fb->frames_base + sizeof(mpi_matrix_frames_t);
}

//

bool
mpi_matrix_framebuffer_push_explicit(
    mpi_matrix_framebuffer_ref  fb,
    base_int_t                  n_elements,
    void                        *elements,
    base_int_t                  *n_elements_remaining
)
{
    base_int_t                  n_elements_rem = n_elements;
    
    if ( ! fb->frames_current || (__mpi_matrix_frame_explicit_get_element_count(fb->datatype, fb->frames_current) < 0) ) {
        // Allocate a new frame:
        if ( __mpi_matrix_framebuffer_alloc_frame(fb, &n_elements_rem) ) {
            // Copy data into the frame:
            memcpy(
                __mpi_matrix_frame_explicit_base_elements_ptr(fb->datatype, fb->frames_current),
                elements,
                __mpi_matrix_frame_delta_byte_size(fb->datatype, (n_elements - n_elements_rem)));
            *n_elements_remaining = n_elements_rem;
            return true;
        }
    }
    else if ( __mpi_matrix_frame_explicit_get_element_count(fb->datatype, fb->frames_current) > 0 ) {
        // Add more element(s) to an explicit frame:
        base_int_t          max_elements = __mpi_matrix_frame_max_elements_allocable(fb->datatype, fb->frames_end - fb->frames_next);
        
        if ( max_elements > 0 ) {
            size_t          byte_delta;
            
            // Fall back to the actual number of elements if the frame buffer has
            // more than sufficient space:
            if ( n_elements < max_elements ) max_elements = n_elements;
            byte_delta = __mpi_matrix_frame_delta_byte_size(fb->datatype, max_elements);
            
            // Copy data into the frame:
            memcpy(
                __mpi_matrix_frame_explicit_end_elements_ptr(fb->datatype, fb->frames_current),
                elements,
                byte_delta);
            fb->frames_next += byte_delta;
            __mpi_matrix_frame_explicit_add_element_count(fb->datatype, fb->frames_current, max_elements);
            *n_elements_remaining = n_elements - max_elements;
            return true;
        }
    }
    *n_elements_remaining = n_elements;
    return false;
}

//

bool
mpi_matrix_framebuffer_push_implicit(
    mpi_matrix_framebuffer_ref  fb,
    base_int_t                  n_elements,
    base_int_t                  *n_elements_remaining
)
{
    base_int_t                  n_elements_rem = -n_elements;
    
    if ( ! fb->frames_current || (__mpi_matrix_frame_explicit_get_element_count(fb->datatype, fb->frames_current) > 0) ) {
        // Allocate a new frame:
        if ( __mpi_matrix_framebuffer_alloc_frame(fb, &n_elements_rem) ) {
            *n_elements_remaining = 0;
            return true;
        }
    }
    else if ( __mpi_matrix_frame_explicit_get_element_count(fb->datatype, fb->frames_current) < 0 ) {
        // Add more element(s) to an explicit frame:
        __mpi_matrix_frame_implicit_add_element_count(fb->datatype, fb->frames_current, n_elements);
        *n_elements_remaining = 0;
        return true;
    }
    *n_elements_remaining = n_elements;
    return false;
}

//

bool
mpi_matrix_framebuffer_is_valid(
    mpi_matrix_framebuffer_ref  fb
)
{
    base_int_t          n_frames = fb->frames_base->n_frames;
    
    if ( n_frames >= 0 ) {
        fb->frames_current = NULL;
        fb->frames_next = (void*)fb->frames_base + sizeof(mpi_matrix_frames_t);
        while ( n_frames-- ) {
            mpi_matrix_frame_implicit_t *cur_frame = (mpi_matrix_frame_implicit_t*)fb->frames_next;
        
            fb->frames_current = fb->frames_next;
            fb->frames_next += __mpi_matrix_frame_base_byte_size(fb->datatype, cur_frame->n_elements);
            if ( fb->frames_next > fb->frames_end ) return false;
        }
        return true;
    }
    return false;
}

//

base_int_t
mpi_matrix_framebuffer_get_frame_count(
    mpi_matrix_framebuffer_ref  fb
)
{
    return fb->frames_base->n_frames;
}

//

base_int_t
mpi_matrix_framebuffer_get_element_count(
    mpi_matrix_framebuffer_ref  fb
)
{
    base_int_t          n_frames = fb->frames_base->n_frames,
                        n_elements = 0;
    void                *frame_ptr = ((void*)fb->frames_base) + sizeof(mpi_matrix_frames_t);
    
    while ( n_frames-- ) {
        n_elements += __base_int_abs(((mpi_matrix_frame_implicit_t*)frame_ptr)->n_elements);
        frame_ptr += __mpi_matrix_frame_base_byte_size(fb->datatype, ((mpi_matrix_frame_implicit_t*)frame_ptr)->n_elements);
    }
    return n_elements;
}

//

const void*
mpi_matrix_framebuffer_iter_next_frame(
    mpi_matrix_framebuffer_ref  fb,
    const void                  *iter_context
)
{
    mpi_matrix_frame_implicit_t *cur_frame;
    
    if ( ! iter_context ) {
        if ( ! fb->frames_base->n_frames ) return NULL;
        return (const void*)fb->frames_base + sizeof(mpi_matrix_frames_t);
    }
    // Advance over the current frame by first testing whether it's implicit or explicit:
    cur_frame = (mpi_matrix_frame_implicit_t*)iter_context;
    iter_context += __mpi_matrix_frame_base_byte_size(fb->datatype, cur_frame->n_elements);
    return ( iter_context >= fb->frames_next ) ? NULL : iter_context;
}

//

base_int_t
mpi_matrix_framebuffer_iter_get_n_elements(
    mpi_matrix_framebuffer_ref  fb,
    const void                  *iter_context
)
{
    return iter_context ? ((mpi_matrix_frame_implicit_t*)iter_context)->n_elements : 0;
}
    
//

bool
mpi_matrix_framebuffer_iter_get_elements_ptr(
    mpi_matrix_framebuffer_ref  fb,
    const void                  *iter_context,
    void*                       *elements_ptr
)
{
    if ( iter_context ) {
        mpi_matrix_frame_implicit_t *cur_frame = (mpi_matrix_frame_implicit_t*)iter_context;
        
        *elements_ptr = (cur_frame->n_elements < 0) ? NULL : __mpi_matrix_frame_explicit_base_elements_ptr(fb->datatype, (void*)iter_context);
        return true;
    }
    return false;
}
