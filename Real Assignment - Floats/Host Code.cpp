#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <CL/cl.hpp>
#include "Utils.h"
#include <chrono>

typedef float mytype;
using namespace std;
using namespace std::chrono;

void print_help() { 
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

//////////////////////////// SUM ///////////////////////////////////////

float Sum(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, vector<mytype> inputData, size_t WorkGroupSize)
{
	size_t workGroupSize = WorkGroupSize;	
	size_t numInputElements = inputData.size();	
	size_t input_size = inputData.size() * sizeof(mytype);//size in bytes	
	vector<mytype> outputData(1);
	size_t output_size = outputData.size() * sizeof(mytype);	

	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);	

	cl::Kernel kernel_sum = cl::Kernel(program, "Sum");
	kernel_sum.setArg(0, inBuffer);
	kernel_sum.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));//local memory size	
	kernel_sum.setArg(2, outBuffer);
			
	
	queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(numInputElements), cl::NDRange(workGroupSize));	

	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);
	
	return (float)outputData[0];
	

}

//////////////////////////// MINIMUM and MAXIMUM ///////////////////////////////////////

float MiniumumMaximum(cl::Context& context, cl::CommandQueue& queue , cl::Program& program, vector<mytype> inputData, size_t WorkGroupSize, int choice)
{
	size_t workGroupSize = WorkGroupSize;	
	size_t numInputElements = inputData.size();	
	size_t input_size = inputData.size() * sizeof(mytype);//size in bytes
	vector<mytype> outputData(1);
	size_t output_size = outputData.size() * sizeof(mytype);

	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);

	cl::Kernel kernel_min = cl::Kernel(program, "MinMax");
	kernel_min.setArg(0, inBuffer);
	kernel_min.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));//local memory size	
	kernel_min.setArg(2, outBuffer);
	kernel_min.setArg(3, choice);


	queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(numInputElements), cl::NDRange(workGroupSize));

	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);

	 return (float)outputData[0];

}

float StandardDeviation(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, vector<mytype> inputData, size_t WorkGroupSize, float mean)
{

	size_t workGroupSize = WorkGroupSize;
	size_t numInputElements = inputData.size();
	size_t input_size = inputData.size() * sizeof(mytype);//size in bytes
	vector<mytype> outputData(1);
	size_t output_size = outputData.size() * sizeof(mytype);

	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);

	cl::Kernel kernel_std = cl::Kernel(program, "SquaredDifferences");
	kernel_std.setArg(0, inBuffer);
	kernel_std.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));//local memory size	
	kernel_std.setArg(2, outBuffer);
	kernel_std.setArg(3, mean);


	queue.enqueueNDRangeKernel(kernel_std, cl::NullRange, cl::NDRange(numInputElements), cl::NDRange(workGroupSize));

	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);	
	return (float)outputData[0];

}


/////////////////////////////// MAIN /////////////////////////////////////

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

//////////////////// Reading the data from the file //////////////////

	ifstream inFileStream("temp_lincolnshire_short.txt");	
//////////////////// OPEN FILE ////////////////////////
	inFileStream.seekg(0, ios_base::end);
	size_t size = inFileStream.tellg();	
	inFileStream.seekg(0, ios_base::beg);	
	char * data = new char[size];
	inFileStream.read(&data[0], size);
//////////////////// CLOSE FILE ///////////////////////
	inFileStream.close();

	//Parse the file by keeping track of the last space before \n
	long spacePos = 0;

	//Final vector of values
	vector<mytype> inputData;

	for (long i = 0; i < size; ++i) 
	{
		char c = data[i];
		if (c == ' ')
		{
			//+1 to i to itterate after last space
			spacePos = i + 1;
		}
		else if (c == '\n')
		{
			int leng = i - spacePos, index = 0;

			//Allocate buffer for word between space_pos and \n
			char * word = new char[leng];
			word[leng] = '\0';

			//Get every char between last space and \n
			for (int j = spacePos; j < i; j++, index++)
			{
				word[index] = data[j];
			}

			//Parse word to double (higher precison)
			inputData.push_back((strtof(word, NULL))*10);
		}
	}
	
///////////////////////////////////////////////////////////////////

	try {

		cl::Context context = GetContext(platform_id, device_id);
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;
		cl::CommandQueue queue(context);
		cl::Program::Sources sources;
		AddSources(sources, "myKernels.cl");
		cl::Program program(context, sources);

		try {
			program.build();
		}
		//display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		
		//Part 4 - memory allocation
		//host - input
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];	
		size_t intialInputSize = inputData.size();
		size_t workGroupSize = 512;
		size_t padding = inputData.size() % workGroupSize;		

		if (padding) 
		{			
			std::vector<mytype> A_ext(workGroupSize - padding, 0);			
			inputData.insert(inputData.end(), A_ext.begin(), A_ext.end());
		}

				
		
		


			
		}
		catch (cl::Error err)
		{
			cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
		}

		return 0;
}




