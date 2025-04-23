#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "quicksort.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 16;
    int *data = NULL;
    int *sub_data = malloc(n / size * sizeof(int));

    if (rank == 0) {
        data = malloc(n * sizeof(int));
        printf("Unsorted array:\n");
        for (int i = 0; i < n; i++) {
            data[i] = rand() % 100;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Scatter(data, n / size, MPI_INT, sub_data, n / size, MPI_INT, 0, MPI_COMM_WORLD);

    quicksort(sub_data, 0, n / size - 1);

    MPI_Gather(sub_data, n / size, MPI_INT, data, n / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Partially sorted array (not fully merged):\n");
        for (int i = 0; i < n; i++) printf("%d ", data[i]);
        printf("\n");
        free(data);
    }

    free(sub_data);
    MPI_Finalize();
    return 0;
}

