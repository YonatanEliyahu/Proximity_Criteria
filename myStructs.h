#pragma once 
/* in general its not commonly used to make an header file for all the structs
   but because I chose to create functions that uses MPI
   there were some linking problems with the MPI and CUDA
   so I chose to do it that way
*/
typedef struct
{
    float x;
    float y;
} Point;

typedef struct{
    float x1;
    float x2;
    float a;
    float b;
} Axis;