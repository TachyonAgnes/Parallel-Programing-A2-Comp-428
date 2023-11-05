#include "mpi.h"
#include <iostream>
#include<math.h>
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

void DisplayResults(int d, int n, double t, std::vector<int> &sortedData) {
	
	std::cout << "\033[32m";  
	std::cout << "\n\n------------------------------------\n";
	std::cout << "Hypercube Dimension: " << d << " \n";
	std::cout << "Input Size: " << n << " \n";
	std::cout << "Parallel execution time: " << t * 1000 * 1000 << " microsecond" << std::endl;
	std::cout << "\033[0m";  
	for (int i = 0; i < sortedData.size(); i++) {
		std::cout << sortedData[i] << " ";
	}
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

	int partitionIndex = 0;
	if(unsortedArray.size() != 0){
		int pivot;
		// Find the median
		int mid = low + (high - low) / 2;
		int pivotValue = medianOfThree(unsortedArray[low], unsortedArray[mid], unsortedArray[high]);

		// We need to ensure that the pivot is at arr[high] for the partition process
		if (pivotValue == unsortedArray[low]) {
			std::swap(unsortedArray[low], unsortedArray[high]);
		}
		else if (pivotValue == unsortedArray[mid]) {
			std::swap(unsortedArray[mid], unsortedArray[high]);
		}
		// Now arr[high] is the pivot element
		pivot = unsortedArray[high];
		// partition the array
		partitionIndex = Partition(unsortedArray, low, high, pivot);
	}
	
	// send the data to the destination
	MPI_Status status;
	int destination = 0;
	// if dimension-th bit is 0
	if ((g_processID & (1 << dimension)) == 0) {
		// prepare data and destination to send
		destination = g_processID + pow(2, dimension);
		if (unsortedArray.size() != 0) {
			dataToSend.assign(unsortedArray.begin() + partitionIndex, unsortedArray.begin() + high + 1);
			// fill the unsorted array with -1
			std::fill(unsortedArray.begin() + partitionIndex, unsortedArray.begin() + high + 1, -1);
			// send and receive data
			MPI_Send(dataToSend.data(), high - partitionIndex + 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
			MPI_Recv(receivedData.data(), receivedData.size(), MPI_INT, destination, 0, MPI_COMM_WORLD, &status);
		}
		else {
			dataToSend.clear();
			MPI_Send(dataToSend.data(), 0, MPI_INT, destination, 0, MPI_COMM_WORLD);
			MPI_Recv(receivedData.data(), receivedData.size(), MPI_INT, destination, 0, MPI_COMM_WORLD, &status);
		}
	}
	// if dimension-th bit is 1
	else {
		// prepare data and destination to send
		destination = g_processID - pow(2, dimension);
		if (unsortedArray.size() != 0) {
			dataToSend.assign(unsortedArray.begin() + low, unsortedArray.begin() + partitionIndex);
			// fill the unsorted array with -1
			std::fill(unsortedArray.begin() + low, unsortedArray.begin() + partitionIndex, -1);
			MPI_Recv(receivedData.data(), receivedData.size(), MPI_INT, destination, 0, MPI_COMM_WORLD, &status);
			MPI_Send(dataToSend.data(), partitionIndex - low, MPI_INT, destination, 0, MPI_COMM_WORLD);
		}
		else {
			dataToSend.clear();
			MPI_Recv(receivedData.data(), receivedData.size(), MPI_INT, destination, 0, MPI_COMM_WORLD, &status);
			MPI_Send(dataToSend.data(), 0, MPI_INT, destination, 0, MPI_COMM_WORLD);
		}
	}

	// merge the data
	std::vector<int> mergedArray;
	mergedArray.reserve(receivedData.size() + unsortedArray.size());
	for (int i = 0; i < receivedData.size(); i++) {
		if (receivedData[i] != -1)
			mergedArray.push_back(receivedData[i]);
	}
	for (int i = 0; i < unsortedArray.size(); i++) {
		if (unsortedArray[i] != -1)
			mergedArray.push_back(unsortedArray[i]);
	}

	return HyperQuickSort(mergedArray, 0, mergedArray.size() - 1, dimension, receivedData, dataToSend);
}

int main(int argc, char *argv[]) {
	double executionTime;
	int arraySize, numOfProcesses, portionSize=0;
	std::vector<int> unsortedArrayBlock, unsortedArray, 
					 buffer, sortedArray, 
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
		if (g_processID == MASTER) {
			unsortedArray = GenerateRandomNumbers(arraySize);
			executionTime = MPI_Wtime();
			SequentialQuickSort(unsortedArray, 0, unsortedArray.size()-1);
			executionTime = MPI_Wtime() - executionTime;
			std::cout << "With arraySize: " << arraySize << ", coreSize: "<< numOfProcesses <<", Sequential execution time: " << executionTime * 1000 * 1000 << " microsecond" << std::endl;
		}
		else {
			unsortedArray.reserve(arraySize);
		}
		
		// initialize vector and parameter
		receivedData.resize(arraySize, -1);
		dataToSend.resize(arraySize);

		// wait for every process to finish
		MPI_Barrier(MPI_COMM_WORLD);
		if (g_processID == MASTER) {
			executionTime = MPI_Wtime();
		}

		// perform hypercube sorting
		std::vector<int> sortedArrayPortion;
		sortedArrayPortion.reserve(arraySize);
		sortedArrayPortion = HyperQuickSort(unsortedArray, 0, unsortedArray.size() - 1,
													  g_dimension, receivedData, dataToSend);

		//// print the result
		//std::stringstream ss;
		//ss << "Process " << g_processID << " has data: ";
		//for (int i = 0; i < sortedArrayPortion.size(); ++i) {
		//	ss << sortedArrayPortion[i] << " ";
		//}
		//ss << std::endl;

		//MPI_Barrier(MPI_COMM_WORLD);

		//std::cout << ss.str();

		// wait for every process to finish
		MPI_Barrier(MPI_COMM_WORLD);
		if (g_processID == MASTER) {
			executionTime = MPI_Wtime() - executionTime;
		}

		sortedArray.reserve(arraySize);
		// merge sorted local result
		if (g_processID != MASTER) {
			MPI_Send(sortedArrayPortion.data(), sortedArrayPortion.size(), MPI_INT, MASTER, 0, MPI_COMM_WORLD); {}
		}
		else {
			// warning, probably have to remove -1 from sorted Array portion, it depends on hyperqs
			sortedArray.insert(sortedArray.end(), sortedArrayPortion.begin(), sortedArrayPortion.end());
			
			// receive data from other process
			for (int i = 1; i < numOfProcesses; i++) {
				// Probe for an incoming message from process i to determine message size
				MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
				// Find out how much data is actually coming
				int count;
				MPI_Get_count(&status, MPI_INT, &count);

				// allocate buffer to receive message
				buffer.resize(count);
				MPI_Recv(buffer.data(), buffer.size(), MPI_INT, i, 0, MPI_COMM_WORLD, &status);
	
				// insert received data into sorted array
				sortedArray.insert(sortedArray.end(), buffer.begin(), buffer.end());
			}

		}
		// display result
		if (g_processID == MASTER) {
		    std::cout << "With arraySize: " << arraySize << ", coreSize: " << numOfProcesses << ", Parallel execution time: " << executionTime * 1000 * 1000 << " microsecond" << std::endl;
			//DisplayResults(g_dimension, arraySize, executionTime, sortedArray);
		}
	}
	MPI_Finalize();

	return 0;
}



