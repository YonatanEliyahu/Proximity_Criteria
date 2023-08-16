#pragma once
#include <mpi.h>
#include <stddef.h>
#include <stdarg.h>
#include "myStructs.h"

void _MPI_severalBcast(int root,MPI_Comm comm,int* counts,MPI_Datatype* types,int numPointers,...); 
void createMPIpoint(MPI_Datatype* point);
void createMPIaxis(MPI_Datatype *axis);