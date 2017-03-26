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

using namespace std;

void print_help() { 
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

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
	typedef float mytype;
	vector<mytype> input_Data;

	string a;
	int b;
	int c;
	int d;
	int e;
	mytype f;

	ifstream infile;
	infile.open("temp_lincolnshire_short.txt");
	if (infile.fail())
	{
		cout << "shits fucked" << endl;
		return 1;
	}

	while (!infile.eof())
	{
		infile >> a >> b >> c >> d >> e >> f;

		input_Data.push_back(f);
	}
	infile.close();

	
	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
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




		//queue.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>
		//(device);
		
		size_t workGroupSize = 512;		
		size_t padding = input_Data.size() % workGroupSize;
		

		if (padding) 
		{
			//create an extra vector with neutral values
			std::vector<mytype> A_ext(workGroupSize - padding, 0);
			//append that extra vector to our input
			input_Data.insert(input_Data.end(), A_ext.begin(), A_ext.end());
		}


			//number of elements
			size_t numInputElements = input_Data.size();
			size_t input_size = input_Data.size() * sizeof(mytype);//size in bytes
			size_t numWorkGroups = numInputElements / workGroupSize;

			//host - output
			vector<mytype> outputData(input_Data.size());
			size_t output_size = numWorkGroups;	
				
			//device - buffers			
			cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
			cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

			//Part 5 - device operations

			//5.1 Copy arrays A and B to device memory
			queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &input_Data[0]);
			queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);//zero B buffer on device memory

			cl::Kernel kernel_sum = cl::Kernel(program, "sumGPU");
			kernel_sum.setArg(0, inBuffer);
			kernel_sum.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));//local memory size			
			kernel_sum.setArg(2, outBuffer);

			queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(numInputElements), cl::NDRange(workGroupSize));

			queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);


			cout << outputData << endl;
		}
		catch (cl::Error err)
		{
			cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
		}

		return 0;
	}
