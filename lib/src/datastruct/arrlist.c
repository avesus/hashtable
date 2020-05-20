#include <datastruct/arrlist.h>

#define PTR_MASK    (((1UL) << 48) - 1)
#define REMOVED_IDX ((arr_node_t *)(~((0UL))))

#define GET_LOG_ISIZE(X) ((X)->mem_addr >> 48)
#define GET_MEM_PTR(X)   ((X)->mem_addr & PTR_MASK)

#define CHECK_REMOVED(X) ((X)->prev == REMOVED_IDX)
#define SET_REMOVED(X)   ((X)->prev = REMOVED_IDX)


#define get_x_idx(X)        ((X)&INT32_MASK)
#define get_y_idx(X)        (((X) >> sizeof_bits(uint32_t)) & INT32_MASK)
#define x_y_to_idx(X, Y, Z) (((X) ? (1 << ((Z) + ((X)-1))) : 0) + (Y))


arr_list_t *
init_alist(uint32_t init_size) {

    const uint64_t _init_size =
        init_size ? roundup_32(init_size) : DEFAULT_INIT_ALSIZE;

    const uint64_t _log_init_size = ulog2_32(_init_size);
    const uint32_t _ntables       = (31 - _log_init_size);
    const uint64_t _max_size =
        sizeof(arr_list_t) + _ntables * sizeof(arr_node_t *) +
        ((1UL) << _log_init_size << _ntables) * sizeof(arr_node_t);

    assert(_max_size > 0);
    assert(_ntables < 32 && _ntables > 0);

    uint64_t start_addr =
        (uint64_t)mymmap(0,
                         _max_size,
                         PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE,
                         (-1),
                         0);


    arr_list_t * new_alist = (arr_list_t *)start_addr;
    start_addr += sizeof(arr_list_t);

    new_alist->arr = (arr_node_t **)(start_addr);
    start_addr += _ntables * sizeof(arr_node_t *);

    new_alist->arr[0] = (arr_node_t *)start_addr;
    start_addr += _init_size * sizeof(arr_node_t);

    start_addr |= _log_init_size << 48;

    new_alist->mem_addr = start_addr;

    return new_alist;
}


void
free_alist(arr_list_t * alist) {
    const uint64_t _log_init_size = GET_LOG_ISIZE(alist);
    const uint32_t _ntables       = (31 - _log_init_size);
    const uint64_t _max_size =
        sizeof(arr_list_t) + _ntables * sizeof(arr_node_t *) +
        ((1UL) << _log_init_size << _ntables) * sizeof(arr_node_t);

    mymunmap(alist, _max_size);
}

static void
add_to_que(arr_list_t * alist, idx_node_t * new_idx_node) {
    new_idx_node->next  = alist->free_idx_que;
    alist->free_idx_que = new_idx_node;
}

static idx_node_t *
check_pop_que(arr_list_t * alist) {
    if (alist->free_idx_que) {
        idx_node_t * ret    = alist->free_idx_que;
        alist->free_idx_que = alist->free_idx_que->next;
        return ret;
    }
    return NULL;
}

static void
add_to_list(arr_list_t * alist, arr_node_t * new_node) {
    new_node->prev = NULL;
    if (alist->ll == NULL) {
        alist->ll      = new_node;
        new_node->next = NULL;
    }
    else {
        new_node->next  = alist->ll;
        alist->ll->prev = new_node;
        alist->ll       = new_node;
    }
}


static void
remove_from_list(arr_list_t * alist, arr_node_t * drop_node) {
    if (drop_node->next == NULL && drop_node->prev == NULL) {
        alist->ll = NULL;
    }
    else if (drop_node->next != NULL && drop_node->prev == NULL) {
        alist->ll       = drop_node->next;
        alist->ll->prev = NULL;
    }
    else if (drop_node->next == NULL && drop_node->prev != NULL) {
        drop_node->prev->next = NULL;
    }
    else {
        drop_node->prev->next = drop_node->next;
        drop_node->next->prev = drop_node->prev;
    }
}


static uint64_t
get_x_y_idx(uint32_t idx, const uint32_t log_init_size) {

    uint32_t x_idx = IDX_TO_X(idx, log_init_size);
    uint32_t y_idx = IDX_TO_Y(idx, x_idx, log_init_size);

    DBG_ASSERT(x_idx < 32, "Error: invalid x index (%d >= %d)\n", x_idx, 32);
    DBG_ASSERT(y_idx < X_TO_SIZE(x_idx, log_init_size),
               "Error: invalid y index (%d >= %d)\n",
               y_idx,
               X_TO_SIZE(x_idx, log_init_size));

    DBG_ASSERT(idx == x_y_to_idx(x_idx, y_idx, log_init_size),
               "Error x_y_to_idx does not match idx\n\t"
               "x_idx   : %d\n\t"
               "y_idx   : %d\n\t"
               "idx     : %d != %d\n",
               x_idx,
               y_idx,
               idx,
               x_y_to_idx(x_idx, y_idx, log_init_size));


    uint64_t ret = y_idx;
    ret <<= sizeof_bits(uint32_t);
    ret |= x_idx;
    return ret;
}


arr_node_t *
add_node(arr_list_t * alist, void * data) {
    arr_node_t * new_node  = NULL;
    idx_node_t * check_idx = check_pop_que(alist);
    if (check_idx) {
        new_node = (arr_node_t *)check_idx;
    }
    else {
        const uint32_t log_init_size = GET_LOG_ISIZE(alist);
        uint32_t       idx           = alist->nitems++;
        uint64_t       x_y_idx       = get_x_y_idx(idx, log_init_size);
        uint32_t       x_idx         = get_x_idx(x_y_idx);
        uint32_t       y_idx         = get_y_idx(x_y_idx);

        if (alist->arr[x_idx] == NULL) {
            if (alist->arr[x_idx] == NULL) {
                alist->arr[x_idx] = (arr_node_t *)GET_MEM_PTR(alist);
                alist->mem_addr +=
                    X_TO_SIZE(x_idx, log_init_size) * sizeof(arr_node_t);
            }
        }
        new_node = alist->arr[x_idx] + y_idx;

        DBG_ASSERT(new_node->data == NULL && new_node->x_idx == 0 &&
                       new_node->y_idx == 0,
                   "Error new node is not new....\n");

        new_node->x_idx = x_idx;
        new_node->y_idx = y_idx;
    }

    new_node->data = data;

    add_to_list(alist, new_node);
    return new_node;
}

arr_node_t *
get_node_idx(arr_list_t * alist, uint32_t idx) {
    const uint32_t log_init_size = GET_LOG_ISIZE(alist);
    uint64_t       x_y_idx       = get_x_y_idx(idx, log_init_size);


    uint32_t x_idx = get_x_idx(x_y_idx);
    uint32_t y_idx = get_y_idx(x_y_idx);

    if (idx >= alist->nitems || alist->arr[x_idx] == NULL ||
        y_idx >= X_TO_SIZE(x_idx, log_init_size)) {
        return NULL;
    }

    arr_node_t * ret_node = alist->arr[x_idx] + y_idx;

    DBG_ASSERT(ret_node->x_idx == x_idx && ret_node->y_idx == y_idx,
               "Error invalid index\n\t"
               "x_idx   : %d vs %d\n\t"
               "y_idx   : %d vs %d\n\t"
               "idx     : %d\n",
               x_idx,
               ret_node->x_idx,
               y_idx,
               ret_node->y_idx,
               idx);

    return CHECK_REMOVED(ret_node) ? NULL : ret_node;
}

void
remove_node_idx(arr_list_t * alist, uint32_t idx) {
    arr_node_t * drop_node = get_node_idx(alist, idx);
    remove_node(alist, drop_node);
}


void
remove_node(arr_list_t * alist, arr_node_t * drop_node) {
    remove_from_list(alist, drop_node);
    SET_REMOVED(drop_node);
    add_to_que(alist, (idx_node_t *)drop_node);
}
