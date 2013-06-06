/*
 * Copyright (c) 2012,
 * Zoltan-Csaba Marton <marton@cs.tum.edu>,
 * Ferenc Balint-Benczedi <balintb.ferenc@gmail.com>,
 * Florian Seidel <seidel.florian@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Intelligent Autonomous Systems Group/
 *       Technische Universitaet Muenchen nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include <tclap/CmdLine.h>
#include <icf_feature_extraction/IMLoaders.hpp>
#include <fstream>
using namespace icf;
using namespace boost;
using namespace std;
using namespace TCLAP;


#define INSTANTIATE_LOADER(P,FE,H) 	fl = FileLoader::Ptr(new FeatureLoader<P,FE,H>())

#define GET_FEATURES(P,F) 	((FeatureLoader<P,F>*) fl.get())->getMatrix()

int main(int argc, char ** argv) {
	//-------------------- Set up command line parser-----------------------------------
	//Command line parser
	CmdLine cmdLine("rospcloader", ' ', "dev");

	//Matcher
	vector<string> availableMatchers;
	availableMatchers.push_back("FlannBased");
	availableMatchers.push_back("BruteForceMatcher");
	availableMatchers.push_back("BruteForce-L1");
	availableMatchers.push_back("BruteForce-HammingLUT");

	ValuesConstraint<string> availableMatchersConstraint(availableMatchers);
	ValueArg<string> matcherArg("","matcher","the name of the matching method",false,"FlannBased",&availableMatchersConstraint);

	ValueArg<int> skipArg("s","skip","number of images to skip between extraction of features",false,1,"int");
	//Feature
//	vector<string> availableDetectors;
//	availableDetectors.push_back("SIFT");
//	availableDetectors.push_back("SURF");
//	availableDetectors.push_back("FAST");
//	availableDetectors.push_back("GFTT");
//	availableDetectors.push_back("MSER");
//	availableDetectors.push_back("STAR");
//	availableDetectors.push_back("HARRIS");
//	availableDetectors.push_back("Grid");
//	availableDetectors.push_back("Pyramid");
//	availableDetectors.push_back("Dynamic");

//	ValuesConstraint<string> availableDetectorsConstraint(availableDetectors);
	ValueArg<string> detectorsArg("d", "detector",
			"[Grid|Pyramid|Dynamic](SIFT|SURF|FAST|GFTT|MSER|STAR|HARRIS)", false, "SURF",
			"string");

	//Feature
	vector<string> availableDescriptors;
	availableDescriptors.push_back("SIFT");
	availableDescriptors.push_back("SURF");
	availableDescriptors.push_back("BRIEF");
	availableDescriptors.push_back("OpponentSURF");
	availableDescriptors.push_back("OpponentSIFT");


	ValuesConstraint<string> availableDescriptorsConstraint(availableDescriptors);
	ValueArg<string> descriptorsArg("f", "feature",
			"The name of a feature  to use", false, "SURF",
			&availableDescriptorsConstraint);

	//base dir
	ValueArg<string> baseDirArg("b", "baseDir",
			"the name of the base folder to be scanned", false, ".", "string");


	//Scale?
	//SwitchArg scaleArg("s", "scale", "scale each feature to [0;1]");

	//name?
	ValueArg<string> nameArg("n", "name", "name of the feature", false, "x",
			"string");

	ValueArg<string> imgExtArg("i","imgext","images end with",true,"_crop.png","string");
	ValueArg<string> maskExtArg("m","mskext","masks end with",false,"_maskcrop.png","string");

	ValueArg<string> cbArg("c","codebook","code book name",true,"cb","string");
	ValueArg<string> cbFileArg("","cbfile","codebookfile",true,"","string");
	ValueArg<string> outArg("o","out","The name of the output file",true,"out","string");

	cmdLine.add(imgExtArg);
	cmdLine.add(maskExtArg);
	cmdLine.add(cbArg);
	cmdLine.add(detectorsArg);
	cmdLine.add(skipArg);
	cmdLine.add(descriptorsArg);
	cmdLine.add(baseDirArg);
	//cmdLine.add(scaleArg);
	cmdLine.add(nameArg);
	cmdLine.add(matcherArg);
	cmdLine.add(outArg);
	cmdLine.add(cbFileArg);
	cmdLine.parse(argc, argv);


	DataSet<double> cb(cbFileArg.getValue());

	Mat codebook = cb.getAsCvMat<double>(cbArg.getValue());

	OpenCVBOWLoader * bl = new OpenCVBOWLoader(imgExtArg.getValue(),maskExtArg.getValue(),detectorsArg.getValue(),descriptorsArg.getValue(),matcherArg.getValue(),codebook,skipArg.getValue());
	FileLoader::Ptr fl(bl);
	HierarchicalPCDLoader loader(fl);

	loader.load(path(baseDirArg.getValue()));
	{
		DataSet<double> out(outArg.getValue(),true,false,16*1024,true);
		out.setFeatureMatrix(bl->getMatrix(),nameArg.getValue());
	}

	return 0;

}
