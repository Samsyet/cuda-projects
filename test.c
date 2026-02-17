#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} Vector;

void vector_init(Vector *v) {
    v->capacity = 1;
    v->size = 0;
    v->data = malloc(v->capacity * sizeof(int));
}

void vector_push(Vector *v, int value) {
    if (v->size == v->capacity) {
        v->capacity *= 2;
        v->data = realloc(v->data, v->capacity * sizeof(int));
    }
    v->data[v->size++] = value;
}

int main() {
    Vector v;
    vector_init(&v);

    vector_push(&v, 10);

    int *ptr = &v.data[0];

    vector_push(&v, 20);

    printf("%d\n", *ptr);

    free(v.data);
    return 0;
} 
