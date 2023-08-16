#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include "myStructs.h"
#include "mpiHelper.h"
#include "myProto.h"

#define MASTER 0

/*
Parallel Computation Final Project
This parallel program utilizes MPI, OpenMP, and CUDA to check the Proximity Criteria for a given set of Axises.
It reads input data from a file, calculates the Proximity Criteria for multiple values of parameter t,
and determines if there exist at least three points satisfying the criteria.
The results are written to an output file.
The program is designed to run faster than the sequential solution by leveraging parallelism and GPU acceleration.
The input file contains the necessary parameters, including the number of Axises,
the minimal number of points for the criteria, the distance threshold, and the value of tCount.
*/

int main(int argc, char *argv[])
{
   // variabels defining
   int size, rank;           // MPI variabels
   int N;                    // size of data
   int K;                    // number of points that define Proximity Criteria
   int tCount;               // Tcount+1 is number of intervals
   float D;                  // D for distance netween points
   Axis *data = NULL;        // arr of axises readed from input file - MASTER ONLY
   Axis *axisChunk = NULL;   // arr of axises (data scattered among processes)
   Point *allPoints = NULL;  // arr of points gathered after PGU calcuation - DEFINED IN ALL PROCESSES
   Point *pointChunk = NULL; // arr of points calculated on GPU (data scattered among processes)
   int *globalFlags = NULL;  // arr of indexes of points that satisfy Proximity Criteria - MASTER ONLY
   int *flags = NULL;        // arr of indexes of points that satisfy Proximity Criteria - per chunk in every process
   double start;             // will hold the starting computation time starting time

   // setting MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // creating MPI Structs that will be knowen in the WORLD
   MPI_Datatype MPI_Axis;
   createMPIaxis(&MPI_Axis);
   MPI_Datatype MPI_Point;
   createMPIpoint(&MPI_Point);

   // reading input file and initalizing neccessary values to run the program
   if (rank == MASTER)
   {
      data = readFile(&N, &K, &D, &tCount);
      if (data == NULL)
         MPI_Abort(MPI_COMM_WORLD, 1);
      if (!printToOutputFile(NULL)) // clear outputflie from last run
      {
         freePointers(1, data); // the following function frees the given pointers only in the processes its defined
         MPI_Abort(MPI_COMM_WORLD, 1);
      }
      start = MPI_Wtime();
   } // end MASTER

   // NOTE: global work scope
   // Bcast neccessary values among MPI_COMM_WORLD
   int counts[4] = {1, 1, 1, 1};
   MPI_Datatype myTypes[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT};
   _MPI_severalBcast(MASTER, MPI_COMM_WORLD, counts, myTypes, 4, &N, &K, &tCount, &D); // the following function will Bcast several pointers
                                                                                       // from the root in the given comm
   // allocating arrs
   int chunkSize = N / size;
   axisChunk = (Axis *)malloc(sizeof(Axis) * chunkSize);
   pointChunk = (Point *)malloc(sizeof(Point) * chunkSize);
   allPoints = (Point *)malloc(N * sizeof(Point));
   if (axisChunk == NULL || pointChunk == NULL || allPoints == NULL)
   {
      freePointers(4, data, axisChunk, pointChunk, allPoints); // the following function frees the given pointers only in the processes its defined
      MPI_Abort(MPI_COMM_WORLD, 2);
   }

   // spliting the axises among MPI_COMM_WORLD
   MPI_Scatter(data, chunkSize, MPI_Axis, axisChunk, chunkSize, MPI_Axis, MASTER, MPI_COMM_WORLD);
   // setting flags arrs for the results of the Proximity Criteria
   flags = (int *)malloc(chunkSize * sizeof(int));
   if (rank == MASTER)
   {
      globalFlags = (int *)malloc(N * sizeof(int));
   }
   int problematicIntervals = 0, i;
   for (i = 0; i <= tCount; i++) // run for all time intervals no limitations // might take a while with some inputs
   {
      double t = 2.0 * i / tCount - 1; // calculate time interval
      if (rank == MASTER)
         printf("i = %d      t = %.02f      ", i, t);
      if (!computePointsOnGPU(axisChunk, pointChunk, chunkSize, t)) // calculate point in the given time
      {
         freePointers(6, data, axisChunk, pointChunk, allPoints, globalFlags, flags); // the following function frees the given pointers only in the processes its defined
         MPI_Abort(MPI_COMM_WORLD, 3);
      }
      MPI_Allgather(pointChunk, chunkSize, MPI_Point, allPoints, chunkSize, MPI_Point, MPI_COMM_WORLD); // gathering all the points to all the processes
      int isLastHits = 0;
      if (i != 0 && rank == MASTER) // APPROVED BONUS SECTION -- checking last points that had been found for time saving
      {
         int res =checkLastHits(allPoints, N, globalFlags, D, K) ;
         isLastHits =  res==3 ? 1 : 0; 
      }
      MPI_Bcast(&isLastHits, 1, MPI_INT, MASTER, MPI_COMM_WORLD); // sending siganl of problematicIntervals to all the slaves to determinate the next steps
      
      if (!isLastHits) // check all point if cound not satesfy by last hits
      {         
         checkProximityCriteriaOnGPU(rank, allPoints, N, flags, chunkSize, D, K);// checking for Proximity Criteria for all the points int the given time
         MPI_Gather(flags, chunkSize, MPI_INT, globalFlags, chunkSize, MPI_INT, MASTER, MPI_COMM_WORLD); // MASTER recive all the calculations
      }

      if (rank == MASTER) // process results of this interval
      {
         int valid = checkFlagsAndPrintOut(N, globalFlags, t); // the following function will check the results and print them (if required) to the output file
         if (valid == -1)                                      // printing went wrong
         {
            freePointers(6, data, axisChunk, pointChunk, allPoints, globalFlags, flags); // the following function frees the given pointers only in the processes its defined
            MPI_Abort(MPI_COMM_WORLD, 4);
         }
         else
         {
            problematicIntervals += valid; // in this case valid is 0 if the weren't at least three points that satisfy the Proximity Criteria
                                           // or 1 if there were at least three points that satisfy the Proximity Criteria
            printf("%s\n", valid ? "found!" : " ");
         }
      }
   }
   if (rank == MASTER)
   {
      double seconds = MPI_Wtime() - start;
      printf("Computation is done - %d Proximity Criteria satisfaction found in %02d:%02d minutes\n avg. time for epoch - %.02f seconds\n",
             problematicIntervals, ((int)seconds) / 60, ((int)seconds) % 60, seconds / i);
      if (problematicIntervals == 0) 
         printToOutputFile((char *)"There were no 3 points found at any t");
   }

   MPI_Type_free(&MPI_Axis);
   MPI_Type_free(&MPI_Point);
   freePointers(6, data, axisChunk, pointChunk, allPoints, globalFlags, flags); // the following function frees the given pointers only in the processes its defined
   MPI_Finalize();
   return 0;
}
