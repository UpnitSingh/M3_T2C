#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x10000)
#define ARRAY_SIZE 16

void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        printf("Error: %s (%d)\n", msg, err);
        exit(1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *data = NULL;
    int chunk_size = ARRAY_SIZE / size;
    int *sub_data = (int*)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        printf("Unsorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % 100;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Scatter(data, chunk_size, MPI_INT, sub_data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // OpenCL setup
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer = NULL;

    cl_int err;

    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

    // Load kernel
    FILE* fp = fopen("quicksort.cl", "r");
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &err);
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "bitonic_sort", &err);

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, chunk_size * sizeof(int), NULL, &err);
    err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, chunk_size * sizeof(int), sub_data, 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), (void*)&chunk_size);

    size_t global_item_size = chunk_size;
    size_t local_item_size = 1;

    cl_event event;
    double start = MPI_Wtime();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
    clFinish(queue);
    double end = MPI_Wtime();

    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, chunk_size * sizeof(int), sub_data, 0, NULL, NULL);

    // Cleanup OpenCL
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    MPI_Gather(sub_data, chunk_size, MPI_INT, data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
        printf("Execution time: %f seconds\n", end - start);
        free(data);
    }

    free(sub_data);
    MPI_Finalize();
    return 0;
}
