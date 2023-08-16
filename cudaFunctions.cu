#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include "myProto.h"
#include "myStructs.h"
#define THREADS 384

__global__ void calculatePoint(Axis *axisArr, Point *pointArr, int numElements, double t)
{ // the following function will calcualte the point location in the specified t
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        pointArr[i].x = ((axisArr[i].x2 - axisArr[i].x1) / 2) * sin(t * M_PI / 2) + (axisArr[i].x2 + axisArr[i].x1) / 2;
        pointArr[i].y = axisArr[i].a * pointArr[i].x + axisArr[i].b;
    }
}
__global__ void SingularProximityCriteria(int index, int *d_temp, Point *pointArr, int N, float D, int K)
{
    // the following function will check if there are K points in the radious of the checked point and set the flag pointer to 1/0 acording to the results
    *d_temp = 0;
    int counter = 0;
    Point p1 = pointArr[index];                // the point we are checking // used for readability
    for (int i = 0; i < N && counter < K; i++) // as long as we didnt go through all the point or we havn't found K points in the radious
    {
        if (index == i)
            continue;
        Point p2 = pointArr[i]; // other point // used for readability
        if (sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) <= D)
            counter++;
    }
    if (counter == K)
        *d_temp = 1;
}

__global__ void ProximityCriteria(int rank, int chunkSize, int *flags, Point *pointArr, int numElements, float D, int K)
{
    // the following function will check if there are K points in the radious of the checked point and will set a flag in the flagArr
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // relative index
    int index = tid + rank * chunkSize;              // acluat index
    int counter = 0;
    if (index < numElements)
    {
        Point p1 = pointArr[index];                          // the point we are checking // used for readability
        for (int i = 0; i < numElements && counter < K; i++) // as long as we didnt go through all the point or we havn't found K points in the radious
        {
            if (index == i)
                continue;
            Point p2 = pointArr[i]; // other point // used for readability
            if (sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) <= D)
                counter++;
        }
        if (counter == K)
            flags[tid] = 1;
    }
}

int computePointsOnGPU(Axis *axisArr, Point *pointArr, int numElements, double t)
{
    /* the following function will get sub arr of axises
        and compute the location of the points in the current t
    */
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // allocate arr for the axises
    size_t size = numElements * sizeof(Axis);
    Axis *d_Axis;
    err = cudaMalloc((void **)&d_Axis, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy data from host to the GPU memory
    err = cudaMemcpy(d_Axis, axisArr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // allocate arr for the points
    size = numElements * sizeof(Point);
    Point *d_Points;
    err = cudaMalloc((void **)&d_Points, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
    int threadsPerBlock = THREADS;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;            // create enough blocks for all the points
    calculatePoint<<<blocksPerGrid, threadsPerBlock>>>(d_Axis, d_Points, numElements, t); // compute points
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy data from GPU to the host memory
    err = cudaMemcpy(pointArr, d_Points, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_Axis);
    cudaFree(d_Points);
    return 1;
}

int checkLastHits(Point *allPoints, int N, int *globalFlags, float D, int K)
{
    /*the following function will calculate the distance between the last hits and the rest of the points,
    if there are K points in a radious of a spesific point (in distancec of D),
    the point will satesfy the Proximity Criteria and will be save in flags arr
    the function will return indicator if those point are enough and the Proximity Criteria is satesfied
    */
    cudaError_t err = cudaSuccess;
    // allocate arr for the points
    size_t size = N * sizeof(Point);
    Point *d_Points;
    err = cudaMalloc((void **)&d_Points, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy data from host to the GPU memory
    err = cudaMemcpy(d_Points, allPoints, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(d_Points);
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int counter = 0;
    int *d_temp;
    err = cudaMalloc((void **)&d_temp, sizeof(int));
    for (int i = 0; i < N && counter < 3; i++) // stops when hits three points that satesfy the Proximity Criteria
    {
        if (globalFlags[i])
        {
            SingularProximityCriteria<<<1, 1>>>(i, d_temp, d_Points, N, D, K); // set singular flag
            cudaMemcpy(&globalFlags[i], d_temp, sizeof(int), cudaMemcpyDeviceToHost);
            if (globalFlags[i])
                counter++;
        }
    }
    cudaFree(d_Points);
    cudaFree(d_temp);
    return counter;
}

void checkProximityCriteriaOnGPU(int rank, Point *allPoints, int N, int *flags, int chunkSize, float D, int K)
{
    /*the following function will calculate the distance between all the points,
    if there are K points in a radious of a spesific point (in distancec of D),
    the point will satesfy the Proximity Criteria and will be save in flags arr (the result of the function)
    */

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    // allocate arr for the points
    size_t size = N * sizeof(Point);
    Point *d_Points;
    err = cudaMalloc((void **)&d_Points, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy data from host to the GPU memory
    err = cudaMemcpy(d_Points, allPoints, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(d_Points);
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Defining flag arr that will hold the points that answer the Proximity Criteria
    int *d_Flags;
    size = chunkSize * sizeof(int);
    err = cudaMalloc((void **)&d_Flags, size);
    if (err != cudaSuccess)
    {
        cudaFree(d_Flags);
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_Flags, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(d_Flags);
        cudaFree(d_Points);
        fprintf(stderr, "Failed to set device memory to zero- %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Launch the Kernel
    int threadsPerBlock = THREADS;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;                                    // create enough blocks for all the points
    ProximityCriteria<<<blocksPerGrid, threadsPerBlock>>>(rank, chunkSize, d_Flags, d_Points, N, D, K); // set flagArr

    err = cudaMemcpy(flags, d_Flags, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(d_Flags);
        cudaFree(d_Points);
        free(flags);
        fprintf(stderr, "Failed to copy data from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_Flags);
    cudaFree(d_Points);
}
