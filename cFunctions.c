#include "myStructs.h"
#include "myProto.h"

int printToOutputFile(char *str) // send NULL to clear output file
{
    FILE *outputStream = fopen(OUTPUTFILE, "w"); // openning output file with write mode to clear the file

    if (outputStream == NULL)
    {
        fprintf(stderr, "can't open %s\n", OUTPUTFILE);
        return 0;
    }
    if (str != NULL)
    {
        fprintf(outputStream, "%s\n", str);
    }

    fclose(outputStream);
    return 1;
}

int printResults(int indexes[3], double t)
{
    FILE *outputStream = fopen(OUTPUTFILE, "a"); // openning output file with append mode

    if (outputStream == NULL)
    {
        fprintf(stderr, "can't open %s\n", OUTPUTFILE);
        return 0;
    }
    int res = fprintf(outputStream, "Points pointID%d, pointID%d, pointID%d satesfy Proximity Criteria at t = %.02f\n", indexes[0], indexes[1], indexes[2], t);
    fclose(outputStream);
    return res;
}

int checkFlagsAndPrintOut(int N, int *globalFlags, double t)
{
    // the following function will check the global flag arr and if 3 points were found, it will be printed to the output file
    int indexes[3] = {-1, -1, -1};
    int counter = 0;
#pragma omp parallel for shared(counter) schedule(dynamic)
    for (int i = 0; i < N; i++)
    {
        if (counter >= 3) // instead of break in serial code
        {
            continue;
        }
        // check if this point flag is on
        if (globalFlags[i] == 1)
        {
#pragma omp critical
            { // avoiding race condition on counter
                if (counter < 3)
                {
                    indexes[counter++] = i;
                }
            }
        }
    }
    if (counter >= 3) // at least 3 problemtic points found
    {
        if (!printResults(indexes, t)) // fail printout -- send -1 as indicator
            return -1;
        else
            return 1;
    }
    return 0;
}

Axis *readFile(int *N, int *K, float *D, int *Tcount)
{
    // the following function will read the input file, set the important values (N,K,D,Tcount)
    //  and return an Axis arr pointer
    printf("reading input file...\n");
    FILE *inputStream = fopen(INPUTFILE, "r"); // openning input file

    if (inputStream == NULL)
    {
        fprintf(stderr, "can't open %s\n", INPUTFILE);
        return NULL;
    }

    if (fscanf(inputStream, "%d %d %f %d\n", N, K, D, Tcount) != 4) // reading critical values for the program
    {
        fclose(inputStream);
        fprintf(stderr, "missing necessary input for the program\n");
        return NULL;
    }
    Axis *inputArr = (Axis *)malloc((*N) * sizeof(Axis)); // allocating the input arr
    if (inputArr == NULL)
    {
        fclose(inputStream);
        fprintf(stderr, "allocation error - couldnwt allocate inputArr\n");
        return NULL;
    }
    for (int i = 0; i < (*N); i++)
    {
        int id;
        if (fscanf(inputStream, "%d ", &id) != 1 || id > *N) // reading the id of the axis // input doesn't must be sorted by id
        {
            fclose(inputStream);
            fprintf(stderr, "reading error - missing index\n");
            free(inputArr);
            return NULL;
        }
        if (fscanf(inputStream, "%f %f %f %f\n", &inputArr[id].x1, &inputArr[id].x2, &inputArr[id].a, &inputArr[id].b) != 4)
        {
            fclose(inputStream);
            fprintf(stderr, "reading error - missing value at index %d\n", id);
            free(inputArr);
            return NULL;
        }
    }
    fclose(inputStream);
    printf("reading went successfully\n");
    return inputArr;
}

void freePointers(int numPointers, ...)
{
    // the following function will get several pointer and free them if they are not null
    va_list pointers;
    va_start(pointers, numPointers);

    for (int i = 0; i < numPointers; i++)
    {
        void *ptr = va_arg(pointers, void *);

        if (ptr != NULL)
        {
            free(ptr);
        }
    }

    va_end(pointers);
}
