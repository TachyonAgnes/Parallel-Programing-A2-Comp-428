#pragma once
#include "mpi.h"
#include <iostream>
#include<math.h>
#include <vector>
#include <iomanip>
#include <sstream>

#define MASTER 0

int g_processID, g_dimension;

MPI_Status status;

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
		int pivot = Partition(vec, low, high, vec[high]);
		SequentialQuickSort(vec, low, pivot - 1);
		SequentialQuickSort(vec, pivot + 1, high);
	}
}

// generate random numbers
std::vector<int> GenerateRandomNumbers(int size) {
	std::vector<int> randomNumbers(size);
	for (int i = 0; i < size; i++) {
		randomNumbers[i] = rand() % size;
	}
	return randomNumbers;
}

void DisplayResults(int d, int n, double t) {
	std::cout << "\033[32m";  
	std::cout << "\n\n------------------------------------\n";
	std::cout << "Hypercube Dimension: " << d << " \n";
	std::cout << "Input Size: " << n << " \n";
	std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << t << " seconds \n";
	std::cout << "\033[0m";  
}

// Hypercube Quick Sort Algo
std::vector<int> HyperQuickSort(std::vector<int> &unsortedArray, int low, int high,
								int dimension, std::vector<int> &receivedData, std::vector<int> &dataToSend) {
	// transfer dimension into zero based indexing
	dimension = dimension - 1; 
	// if recursion depth is equal to d, stop the recursion
	if (dimension == -1) {
		SequentialQuickSort(unsortedArray, low, high);
		//return as sorted
		return unsortedArray;
	}

	// initialize receive and merged buffer
	std::fill(receivedData.begin(), receivedData.end(), -1);

	int pivot;
	pivot = unsortedArray[high];
	int index = unsortedArray.size() - 1;
	while (pivot <= 0 && index >= 0) {
		pivot = unsortedArray[index];
		index--;
	}

	// partition the array
	int partitionIndex = Partition(unsortedArray, low, high, pivot);

	std::cout << pivot << std::endl;

	// send the data to the destination
	MPI_Status status;
	int destination = 0;
	// if dimension-th bit is 0
	if ((g_processID & (1 << dimension)) == 0) {
		// prepare data and destination to send
		destination = g_processID + std::pow(2, dimension);
		dataToSend.assign(unsortedArray.begin() + partitionIndex, unsortedArray.begin() + high + 1);
		// fill the unsorted array with -1
		std::fill(unsortedArray.begin() + partitionIndex, unsortedArray.begin() + high + 1, -1);

		// send and receive data
		MPI_Send(dataToSend.data(), high - partitionIndex + 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
		MPI_Recv(receivedData.data(), receivedData.size(), MPI_INT, destination, 0, MPI_COMM_WORLD, &status);
	}
	// if dimension-th bit is 1
	else {
		// prepare data and destination to send
		destination = g_processID - std::pow(2, dimension);
		dataToSend.assign(unsortedArray.begin() + low, unsortedArray.begin() + partitionIndex);
		// fill the unsorted array with -1
		std::fill(unsortedArray.begin() + low, unsortedArray.begin() + partitionIndex, -1);

		MPI_Recv(receivedData.data(), receivedData.size(), MPI_INT, destination, 0, MPI_COMM_WORLD, &status);
		MPI_Send(dataToSend.data(), partitionIndex - low, MPI_INT, destination, 0, MPI_COMM_WORLD);
	}

	// merge the data
	std::vector<int> mergedArray;
	mergedArray.reserve(receivedData.size() + unsortedArray.size());
	for (int i = 0; i < receivedData.size(); i++) {
		if(receivedData[i] != -1)
			mergedArray.push_back(receivedData[i]);
	}
	for (int i = 0; i < unsortedArray.size(); i++) {
		if(unsortedArray[i] != -1)
			mergedArray.push_back(unsortedArray[i]);
	}
	mergedArray.shrink_to_fit();

	return HyperQuickSort(mergedArray, 0, mergedArray.size() - 1,
						  dimension, receivedData, dataToSend);
}

int main(int argc, char *argv[]) {
	double executionTime;
	int arraySize, numOfProcesses, portionSize=0;
	std::vector<int> unsortedArrayBlock, unsortedArray, 
					 buffer, finalMergedArray, 
					 receivedData, mergedArray, dataToSend;

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
		
		// we are going to spread data equally into each cpu
		// portionSize stands for smallest size of each portion
		portionSize = arraySize / numOfProcesses;
		int remainder = arraySize % numOfProcesses;
		std::vector<int> sendCounts(numOfProcesses, portionSize);
		std::vector<int> displs(numOfProcesses, 0);

		if (g_processID == MASTER) {
			unsortedArray = GenerateRandomNumbers(arraySize);
			
			for (int i = 0; i < remainder; ++i) {
				sendCounts[i]++;
			}
			for (int i = 1; i < numOfProcesses; ++i) {
				displs[i] = displs[i - 1] + sendCounts[i - 1];
			}
		}
		MPI_Bcast(sendCounts.data(), sendCounts.size(), MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(displs.data(), displs.size(), MPI_INT, MASTER, MPI_COMM_WORLD);
		unsortedArrayBlock.resize(sendCounts[g_processID]);

		// send the portion size and unsorted array to all process
		MPI_Scatterv(unsortedArray.data(), sendCounts.data(), displs.data(), MPI_INT, unsortedArrayBlock.data(), sendCounts[g_processID], MPI_INT, MASTER, MPI_COMM_WORLD);
		
		// initialize vector and parameter
		receivedData.resize(arraySize, -1);
		dataToSend.resize(arraySize);

		//finalMergedArray.resize(arraySize);
		//buffer.resize(arraySize);

		// wait for every process to finish
		MPI_Barrier(MPI_COMM_WORLD);
		if (g_processID == MASTER) {
			executionTime = MPI_Wtime();
		}

		// perform hypercube sorting
		std::vector<int> sortedArray(sendCounts[g_processID], g_processID);
		sortedArray = HyperQuickSort(unsortedArrayBlock, 0, sendCounts[g_processID] - 1,
													  g_dimension, receivedData, dataToSend);

		///  tester
		std::stringstream ss;
		ss << "Process " << g_processID << " has data: ";
		for (int i = 0; i < sortedArray.size(); ++i) {
			ss << sortedArray[i] << " ";
		}
		ss << std::endl;

		MPI_Barrier(MPI_COMM_WORLD);

		std::cout << ss.str();
		/// end of tester

		MPI_Barrier(MPI_COMM_WORLD);

		//// wait for every process to finish
		//MPI_Barrier(MPI_COMM_WORLD);
		//if (g_processID == MASTER) {
		//	executionTime = MPI_Wtime() - executionTime;
		//}

		//// merge sorted local result
		//if (g_processID != MASTER) {
		//	MPI_Send(sortedArray.data(), portionSize, MPI_INT, MASTER, 0, MPI_COMM_WORLD); {}
		//}
		//else {
		//	// warning, probably have to remove -1 from sorted Array, it depends on hyperqs
		//	finalMergedArray.insert(finalMergedArray.end(), sortedArray.begin(), sortedArray.end());
		//	
		//	// receive data from other process
		//	for (int i = 1; i < numOfProcesses; i++) {
		//		MPI_Recv(buffer.data(), portionSize, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
		//		// warning, probably have to remove -1 from sorted Array, it depends on hyperqs
		//		/*end = std::remove(buffer.begin(), buffer.end(), -1);*/
		//		finalMergedArray.insert(finalMergedArray.end(), buffer.begin(), buffer.end());
		//	}

		//}
		//// display result
		//if (g_processID == MASTER) {
		//	DisplayResults(g_dimension, arraySize, executionTime);
		//}
	}
	MPI_Finalize();

	return 0;
}



