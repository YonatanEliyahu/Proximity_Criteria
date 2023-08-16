build:
	mpicxx -fopenmp -Wall -Wextra -c main.c -o main.o
	mpicxx -fopenmp -Wall -Wextra -c mpiHelper.c -o mpiHelper.o
	mpicxx -fopenmp -Wall -Wextra -c cFunctions.c -o cFunctions.o
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -Xcompiler -Wall -Xcompiler -Wextra -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  cFunctions.o cudaFunctions.o mpiHelper.o main.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart -lmpi
	

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP
