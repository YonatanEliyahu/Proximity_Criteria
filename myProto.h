#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "myStructs.h"

#define INPUTFILE "./Input.txt"
#define OUTPUTFILE "./Output.txt"


int computePointsOnGPU(Axis* axisChunk,Point* pointChunk, int chunkSize, double t);
void checkProximityCriteriaOnGPU(int rank, Point *allPoints, int N,int* flags, int chunkSize, float D, int K);
int checkLastHits(Point *allPoints, int N, int *globalFlags, float D, int K);

Axis* readFile(int* N, int* K, float* D, int* Tcount);

int printToOutputFile(char* str);
int printResults(int indexes[3],double t);
int checkFlagsAndPrintOut(int N, int *globalFlags, double t);

void freePointers(int numPointers, ...);
