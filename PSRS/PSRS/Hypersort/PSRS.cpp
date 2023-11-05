#include "mpi.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cmath>

#define MASTER 0

int g_processID, g_dimension;

MPI_Status status;

int medianOfThree(int a, int b, int c) {
	if ((a - b) * (c - a) >= 0) {
		return a; // a is the median
	}
	else if ((b - a) * (c - b) >= 0) {
		return b; // b is the median
	}
	else {
		return c; // c is the median
	}
}

// partition
int Partition(std::vector<int> &vec, int low, int high, int pivotValue) {
	int smallerElementIndex = low - 1;

	for (int i = low; i < high; i++) {
		if (vec[i] <= pivotValue) {
			smallerElementIndex++;
			std::swap(vec[smallerElementIndex], vec[i]);
		}
	}
	std::swap(vec[smallerElementIndex + 1], vec[high]);
	return (smallerElementIndex + 1);
}

// sequential quick sort
void SequentialQuickSort(std::vector<int> &vec, int low, int high) {
	if (low < high) {
		// Find the median among arr[low], arr[high], and arr[(low+high)/2]
		int mid = low + (high - low) / 2;
		int pivotValue = medianOfThree(vec[low], vec[mid], vec[high]);

		// We need to ensure that the pivot is at arr[high] for the partition process
		if (pivotValue == vec[low]) {
			std::swap(vec[low], vec[high]);
		}
		else if (pivotValue == vec[mid]) {
			std::swap(vec[mid], vec[high]);
		}
		// Now arr[high] is the pivot element

		int pivot = Partition(vec, low, high, vec[high]);
		SequentialQuickSort(vec, low, pivot - 1);
		SequentialQuickSort(vec, pivot + 1, high);
	}
}

// generate random numbers
std::vector<int> GenerateRandomNumbers(int size) {
	//srand(static_cast<unsigned int>(std::time(nullptr)));
	srand(42);
	std::vector<int> randomNumbers(size);
	for (int i = 0; i < size; i++) {
		randomNumbers[i] = rand() % size;
	}
	return randomNumbers;
}

void DisplayResults(int p, int n, double t, std::vector<int> &sortedData) {
	
	std::cout << "\033[32m";  
	std::cout << "\n\n------------------------------------\n";
	std::cout << "Core Size: " << p << " \n";
	std::cout << "Input Size: " << n << " \n";
	std::cout << "Parallel execution time: " << t * 1000 * 1000 << " microsecond" << std::endl;
	std::cout << "\033[0m";  
	for (int i = 0; i < sortedData.size(); i++) {
		std::cout << sortedData[i] << " ";
	}
}

int main(int argc, char *argv[]) {
	double executionTime = 0;
	int arraySize, numOfProcesses, arrayBlockSize;
	std::vector<int> unsortedArray, unsortedArrayBlock, sortedArray;

	int elementsCounter = 0;

	// Initialize MPI
	MPI_Init(&argc, &argv);

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

	// Get the ID of this process
	MPI_Comm_rank(MPI_COMM_WORLD, &g_processID);

	// Get the number of dimensions
	g_dimension = log2(numOfProcesses);

	// arguments must be entered 
	if (argc < 2) {
		if (g_processID == MASTER){
			std::cout << "\033[1;31m" << std::endl
				<< "ERROR: NUMBER OF PROCESSES MUST BE LESS THAN n" 
				<< "\nFORMAT: mpirun - np <numOfProcesses> Par <arraySize>" << std::endl
				<< "\033[0m";
		}
	}

	// The input size of the n must be greater than the number of process
	else if (atoi(argv[1]) < numOfProcesses) {
		if (g_processID == MASTER) {
			std::cout << "\033[1;31m \nERROR: NUMBER OF PROCESSES MUST BE LESS THAN n\n\n \033[0m" << std::endl;
		}
	}

	// The number of processes must be a power of 2
	else if (numOfProcesses != pow(2, g_dimension)) {
		if (g_processID == MASTER){
			std::cout << "\033[1;31m" << std::endl
				<< "ERROR: NUMBER OF PROCESSES MUST BE A POWER OF 2" << std::endl << std::endl
				<< "\033[0m";
		}
	}
	else {
		// Input size
		arraySize = atoi(argv[1]);

		/// Step 1 using MPI_Scatterv to distribute the data into the processes
		// if there are remainder, the local size will be increased by 1
		int portionSize = arraySize / numOfProcesses;
		int remainder = arraySize % numOfProcesses;
		// allocate memory for the local array
		unsortedArrayBlock.resize(portionSize);

		std::vector<int> masterExtraData;
		if (g_processID == MASTER) {
			// Generate random numbers
			unsortedArray = GenerateRandomNumbers(arraySize);
			// sort the data using sequential quick sort
			executionTime = MPI_Wtime();
			SequentialQuickSort(unsortedArray, 0, unsortedArray.size() - 1);
			executionTime = MPI_Wtime() - executionTime;
			std::cout << "With arraySize: " << arraySize << ", coreSize: " << numOfProcesses << ", Sequential execution time: " << executionTime * 1000 * 1000 << " microsecond" << std::endl;

			// start the timer
			if (g_processID == MASTER) {
				executionTime = MPI_Wtime();
			}
			masterExtraData.reserve(remainder);
			// distribute the data into the processes
			masterExtraData.insert(masterExtraData.end(),
								   unsortedArray.begin() + portionSize * numOfProcesses,
								   unsortedArray.end());

			// resize the array
			unsortedArray.erase(unsortedArray.begin() + portionSize * numOfProcesses, unsortedArray.end());
		}

		
		MPI_Scatter(unsortedArray.data(), portionSize, MPI_INT, 
					unsortedArrayBlock.data(), portionSize, MPI_INT, MASTER, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		// if master
		if (g_processID == MASTER) {
			unsortedArrayBlock.insert(unsortedArrayBlock.end(), masterExtraData.begin(), masterExtraData.end());
		}

		/// Step 2 using MPI_Gather collects all selected sample value from all processes to the master process

		// calculate sampling interval
		arrayBlockSize = unsortedArrayBlock.size();
		int localSampleCount = numOfProcesses;
		int samplingInterval = arrayBlockSize / localSampleCount;
		// initiate an array to store the sample values
		std::vector<int> sampleValues(localSampleCount);

		// select the sample values
		for (int i = 0; i < localSampleCount; ++i) {
			sampleValues[i] = unsortedArrayBlock[i * samplingInterval];
		}
		
		// if there are remainder, the last element of the array will be the last element of the block
		if (localSampleCount > 1) {
			sampleValues[localSampleCount - 1] = unsortedArrayBlock[arrayBlockSize - 1];
		}

		// initialize an array to store all sample values
		std::vector<int> allSampleValues;
		if (g_processID == MASTER) {
			allSampleValues.resize(numOfProcesses * numOfProcesses);
		    // the master process uses allSampleValues as the receiving buffer
			MPI_Gather(sampleValues.data(), numOfProcesses, MPI_INT, allSampleValues.data(),
				numOfProcesses, MPI_INT, MASTER, MPI_COMM_WORLD);
		}
		else {
			// the other processes use sampleValues as the sending buffer
			MPI_Gather(sampleValues.data(), numOfProcesses, MPI_INT, NULL,
				0, MPI_INT, MASTER, MPI_COMM_WORLD);
		}

		/// Step 3 using MPI_Bcast to broadcast the selected sample values to all processes
		std::vector<int> selectedPivot(numOfProcesses - 1);
		if(g_processID == MASTER) {
			// quick sort the sample values
			SequentialQuickSort(allSampleValues, 0, allSampleValues.size() - 1);
			// select the sample values
			for (int i = 0; i < numOfProcesses - 1; ++i) {
				selectedPivot[i] = allSampleValues[i * numOfProcesses + numOfProcesses / 2 - 1];
			}
			// broadcast the sample values to all processes
			MPI_Bcast(selectedPivot.data(), numOfProcesses, MPI_INT, MASTER, MPI_COMM_WORLD);
		}
		else {
			// receive the sample values from the master process
			MPI_Bcast(selectedPivot.data(), numOfProcesses, MPI_INT, MASTER, MPI_COMM_WORLD);
		}
		/// Step 4 using MPI_AlltoALLv to distribute the data into the processes


		std::vector<int> sendbuf;
		sendbuf.reserve(arrayBlockSize);
		std::vector<int> sendCounts(numOfProcesses);
		std::vector<int> sdispls(numOfProcesses);

		// partition the unsortedArrayBlock by selectedPivot
		std::vector<std::vector<int>> partitions(numOfProcesses);
		for (int i = 0; i < unsortedArrayBlock.size(); ++i) {
			// find the partition index
			int partitionIndex = std::upper_bound(selectedPivot.begin(), selectedPivot.end(), 
												  unsortedArrayBlock[i]) - selectedPivot.begin();
			// add the element to the partition
			partitions[partitionIndex].push_back(unsortedArrayBlock[i]);
		}

		// calculate the sendCounts and sdispls
		int displacement = 0;
		for (int i = 0; i < numOfProcesses; ++i) {
			sendCounts[i] = partitions[i].size();
			sdispls[i] = displacement;
			displacement += sendCounts[i];
			sendbuf.insert(sendbuf.end(), partitions[i].begin(), partitions[i].end());
		}

		std::vector<int> recvbuf;
		std::vector<int> recvCounts(numOfProcesses);
		std::vector<int> rdispls(numOfProcesses);

		MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
					 recvCounts.data(), 1, MPI_INT,
					 MPI_COMM_WORLD);

		rdispls[0] = 0;
		int totalRecvSize = recvCounts[0];
		for (int i = 1; i < numOfProcesses; ++i) {
			rdispls[i] = rdispls[i-  1] + recvCounts[  i-1];
			totalRecvSize += recvCounts[i];
		}
		recvbuf.resize(totalRecvSize);

		MPI_Barrier(MPI_COMM_WORLD);
		
		MPI_Alltoallv(sendbuf.data(), sendCounts.data(), sdispls.data(), MPI_INT,
					  recvbuf.data(), recvCounts.data(), rdispls.data(), MPI_INT,
					  MPI_COMM_WORLD);

		// print the result
		MPI_Barrier(MPI_COMM_WORLD);
		/// Step 5 collect and merge the data from all processes to the master process

		// local quick sort
		SequentialQuickSort(recvbuf, 0, recvbuf.size() - 1);

		sortedArray.reserve(arraySize);
		// merge sorted local result
		if (g_processID != MASTER) {
			MPI_Send(recvbuf.data(), recvbuf.size(), MPI_INT, MASTER, 0, MPI_COMM_WORLD); {}
		}
		else {
			// warning, probably have to remove -1 from sorted Array portion, it depends on hyperqs
			sortedArray.insert(sortedArray.end(), recvbuf.begin(), recvbuf.end());

			// receive data from other process
			for (int i = 1; i < numOfProcesses; i++) {
				// Probe for an incoming message from process i to determine message size
				MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
				// Find out how much data is actually coming
				int count;
				MPI_Get_count(&status, MPI_INT, &count);

				// allocate buffer to receive message
				recvbuf.resize(count);
				MPI_Recv(recvbuf.data(), recvbuf.size(), MPI_INT, i, 0, MPI_COMM_WORLD, &status);

				// insert received data into sorted array
				sortedArray.insert(sortedArray.end(), recvbuf.begin(), recvbuf.end());
			}

		}
		if (g_processID == MASTER) {
			executionTime = MPI_Wtime() - executionTime;
			std::cout << "With arraySize: " << arraySize << ", coreSize: " << numOfProcesses << ", Parallel execution time: " << executionTime * 1000 * 1000 << " microsecond" << std::endl;
		}

	}
	/// Step 7 MPI_finalize
	MPI_Finalize();

	return 0;
}



