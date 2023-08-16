#include "mpiHelper.h"

void _MPI_severalBcast(int root,MPI_Comm comm,int* counts,MPI_Datatype* types,int numPointers ,...){
    //the following function will Bcast several pointers from the root to the other processes int the MPI_Comm
    va_list pointers;
    va_start(pointers, numPointers);
    for(int i=0;i<numPointers;i++){
        void* ptr = va_arg(pointers, void*);
        MPI_Bcast( ptr , counts[i] , types[i] , root , comm);
    }
    va_end(pointers);
}

void createMPIpoint(MPI_Datatype *point)
{
    //the following function will create mpi type of Point an set the pointer to hold that type
    MPI_Datatype type[2] = { MPI_FLOAT, MPI_FLOAT};
    int blocklen[2]={1,1};
    MPI_Aint disp[2];

    disp[0]= offsetof(Point,x);
    disp[1]= offsetof(Point,y);
    MPI_Type_create_struct(2,blocklen,disp,type,point);
    MPI_Type_commit(point);
}

void createMPIaxis(MPI_Datatype *axis)
{
    // the following function will create an mpi_axis type an set the pointer to hold that type
    MPI_Datatype type[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    int blocklen[4] = {1, 1, 1, 1};
    MPI_Aint disp[4];

    disp[0] = offsetof(Axis, x1);
    disp[1] = offsetof(Axis, x2);
    disp[2] = offsetof(Axis, a);
    disp[3] = offsetof(Axis, b);
    MPI_Type_create_struct(4, blocklen, disp, type, axis);
    MPI_Type_commit(axis);
}