#ifndef _ARR_LIST_H_
#define _ARR_LIST_H_

#include <general-config.h>

#include <helpers/debug.h>
#include <helpers/opt.h>
#include <helpers/util.h>


#define DEFAULT_INIT_ALSIZE (128)


typedef struct arr_node {
    uint32_t          x_idx;
    uint32_t          y_idx;
    struct arr_node * next;
    struct arr_node * prev;
    void *            data;
} arr_node_t;

typedef struct idx_node {
    uint32_t          x_idx;
    uint32_t          y_idx;
    struct idx_node * next;
} idx_node_t;

typedef struct arr_list {
    arr_node_t ** arr;
    arr_node_t *  ll;
    idx_node_t *  free_idx_que;
    uint64_t      mem_addr;
    uint32_t      nitems;
} arr_list_t;


arr_list_t * init_alist(uint32_t total_items);
void         free_alist(arr_list_t * a_list);

void remove_node_idx(arr_list_t * alist, uint32_t idx);
void remove_node(arr_list_t * alist, arr_node_t * drop_node);

arr_node_t * add_node(arr_list_t * alist, void * data);

arr_node_t * get_node_idx(arr_list_t * alist, uint32_t idx);

#endif
