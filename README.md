# Parallel Implementation of Proximity Criteria

This repository contains the final project for Course 10324, Parallel and Distributed Computation, Fall Semester 2023. The project focuses on the parallel implementation of the Proximity Criteria algorithm using a combination of MPI, OpenMP, and CUDA. The goal is to efficiently find points that satisfy the Proximity Criteria for a given set of axises.

## Problem Definition

The task involves analyzing a set of N points in a two-dimensional plane. The coordinates of each point are defined by the equations:
x = ((x2 - x1) / 2) * sin(t * π / 2) + ((x2 + x1) / 2)
y = a * x + b
where (x1, x2, a, b) are predefined constant parameters for each point.

## Requirements
- Perform Proximity Criteria checks for tCount + 1 values of t, where t = 2 * i / tCount - 1, and i ranges from 0 to tCount.
- For each value of t, identify if there are at least three points that satisfy the Proximity Criteria. If found, no further evaluation is needed for that specific value of t.
- Read input from input.txt and write results to output.txt.
- Ensure the parallel program's computation time is faster than a sequential solution.
- Be prepared to demonstrate the solution on VLAB with MPI involving two computers from different pools.
- The project must include comprehensive explanations of each line of code, even those reused from other sources.
Projects with missing files, build/run errors, or improper creation will not be accepted.

## Input and Output
### Input
The input.txt file contains the following data:
N K D TCount
id x1 x2 a b
id x1 x2 a b
id x1 x2 a b
...
id x1 x2 a b

where N is the number of points, K is the minimal number of points to satisfy the Proximity Criteria, D is the distance threshold, and TCount is the value used to calculate t. Each point has an ID (id), constants x1, x2, a, and b.

### Output
The output.txt file contains information about points that satisfy the Proximity Criteria for specific values of t:
Points pointID1, pointID2, pointID3 satisfy Proximity Criteria at t = t1
Points pointID4, pointID5, pointID6 satisfy Proximity Criteria at t = t2
Points pointID7, pointID8, pointID9 satisfy Proximity Criteria at t = t3
If no points are found for any t, the output will be:
There were no 3 points found for any t.

## Implementation Overview
The parallel program utilizes MPI, OpenMP, and CUDA to efficiently check the Proximity Criteria for a given set of axises. It processes the input data, calculates the criteria for multiple t values, and determines if there are at least three points satisfying the criteria. The goal is to outperform the sequential solution by leveraging parallelism and GPU acceleration.

For more detailed explanations of the code, refer to the source code files and accompanying documentation.

## Instructions
1. Clone this repository to your local machine.
2. Compile and execute the parallel program.
3. Ensure the results match the expected output format.
   
Feel free to reach out for further clarifications or assistance.

