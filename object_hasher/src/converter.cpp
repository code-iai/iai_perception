/*
 * converter.cpp
 *
 *  Created on: Jan 17, 2013
 *      Author: ferenc
 */


#include <iostream>
#include <icf_dataset/DataSet.hpp>
#include <stdio.h>
#include <fstream>

#include <tclap/CmdLine.h>
#include <boost/algorithm/string.hpp>

using namespace TCLAP;


void splitNonEmpty(std::vector<std::string> &values,  std::string input,std::string is_any_of)
{
	boost::split (values, input, boost::is_any_of (is_any_of), boost::token_compress_on);
	std::vector<std::string>::iterator it = values.begin ();
	while (it != values.end())
	{
		if (*it == "")
			it = values.erase(it);
		else
			it++;
	}
}

int main(int argc, char** argv)
{
	//Command line parser
	std::cerr<<"STARTED CONCAT!"<<std::endl;
	CmdLine cmdLine("converter", ' ', "dev");

	ValueArg<std::string> inFile("i","input_file","path to old style input file",true, "x1","string");
	ValueArg<std::string> outFile("o","output_file","path to new file",true, "x2","string");

	cmdLine.add(inFile);
	cmdLine.add(outFile);
	cmdLine.parse(argc,argv);

	std::ifstream infile(inFile.getValue().c_str());
	icf::DataSet<double> ds;
	icf::DS::Matrix extractedFeatures;
	icf::DS::Matrix lMatrix;
	icf::DS::Matrix aMatrix;
	icf::DS::Matrix pMatrix;

	bool first_run=true;
	int nr=0;
	if (infile.is_open())
	{
		std::cerr<<"[note] succesfully opened file "<<std::endl;
		std::string line;
		while (std::getline(infile, line))
		{
			std::vector<std::string> values;
			splitNonEmpty(values,line, ", ()");
			if (first_run)
			{
				extractedFeatures=icf::DS::Matrix(1, values.size()-3);
				lMatrix=icf::DS::Matrix(1,1);
				aMatrix=icf::DS::Matrix(1,1);
				pMatrix=icf::DS::Matrix(1,1);
			}
			else
			{
				extractedFeatures.conservativeResize(extractedFeatures.rows()+1,extractedFeatures.cols());
				lMatrix.conservativeResize(lMatrix.rows()+1,lMatrix.cols());
				aMatrix.conservativeResize(aMatrix.rows()+1,aMatrix.cols());
				pMatrix.conservativeResize(pMatrix.rows()+1,pMatrix.cols());
			}

			pMatrix(nr,0) = atoi(values.at(0).c_str());
			aMatrix(nr,0) =	atoi(values.at(1).c_str());
			lMatrix(nr,0) = atoi(values.at(2).c_str());
			for(unsigned int j = 3;j!=values.size();j++)
				extractedFeatures(nr,j-3)=atof(values.at(j).c_str());
			nr++;
			first_run=false;
		}
		infile.close();
		std::cerr<<"[note] succesfully read file to stream...adding as training data"<<std::endl;
	}

	ds.setFeatureMatrix(extractedFeatures, "x");
	ds.setFeatureMatrix(lMatrix,"y");
	ds.setFeatureMatrix(aMatrix,"arr");
	ds.setFeatureMatrix(pMatrix,"partNr");

	if (!outFile.isSet())
	{
		std::cout << ds << std::flush;
	}
	else
	{
		std::fstream out;
		out.open(outFile.getValue().c_str(), std::ios_base::out);
		out << ds << std::flush;
	}
	return 0;
}

