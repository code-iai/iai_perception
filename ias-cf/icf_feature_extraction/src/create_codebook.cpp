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
//
//	ValuesConstraint<string> availableDetectorsConstraint(availableDetectors);
	ValueArg<string> detectorsArg("d", "detector",
				"[Grid|Pyramid|Dynamic](SIFT|SURF|FAST|GFTT|MSER|STAR|HARRIS)", false, "SURF",
				"string");
	ValueArg<int> skipArg("s","skip","number of images to skip between extraction of features",false,1,"int");
	//Feature
	vector<string> availableDescriptors;
	availableDescriptors.push_back("SIFT");
	availableDescriptors.push_back("SURF");
	availableDescriptors.push_back("BRIEF");
	availableDescriptors.push_back("OpponentSURF");
	availableDescriptors.push_back("OpponentSIFT");

	ValuesConstraint<string> availableDescriptorsConstraint(availableDescriptors);
	ValueArg<string> descriptorsArg("f", "feature",
			"Then name of a feature  to use", false, "SURF",
			&availableDescriptorsConstraint);

	//base dir
	ValueArg<string> baseDirArg("b", "baseDir",
			"the name of the base folder to be scanned", false, ".", "string");
	//Scale?
	//SwitchArg scaleArg("s", "scale", "scale each feature to [0;1]");

	//name?
	ValueArg<string> nameArg("n", "name", "name of the feature", false, "x",
			"string");

	ValueArg<string> outArg("o","out","The name of the output file",true,"out","string");

	ValueArg<int> kArg("c","clusters","Number of code book vectors",true,600,"int>0");

	ValueArg<int> retriesArg("r","retries","kmeans retries",false,1,"int>0");

	ValueArg<string> imgExtArg("i","imgext","images end with",false,"_crop.png","int>0");
	ValueArg<string> maskExtArg("m","mskext","masks end with",false,"_maskcrop.png","int>0");

	cmdLine.add(imgExtArg);
	cmdLine.add(maskExtArg);
	cmdLine.add(retriesArg);
	cmdLine.add(kArg);
	cmdLine.add(detectorsArg);
	cmdLine.add(descriptorsArg);
	cmdLine.add(baseDirArg);
	cmdLine.add(skipArg);
	cmdLine.add(nameArg);
	cmdLine.add(outArg);
	cmdLine.parse(argc, argv);

	OpenCVFeatureLoader* ofl = new OpenCVFeatureLoader(imgExtArg.getValue(),maskExtArg.getValue(),detectorsArg.getValue(),descriptorsArg.getValue(),skipArg.getValue());
	FileLoader::Ptr fl(ofl);

	HierarchicalPCDLoader hloader(fl);

	hloader.load(path(baseDirArg.getValue()));

	OpenCVFeatureLoader::LocalFeatures features = ofl->getLocalFeatures();
	cout<<features.size()<<endl;
	BOWKMeansTrainer trainer(kArg.getValue(),TermCriteria(CV_TERMCRIT_ITER,100,0.001),retriesArg.getValue(),KMEANS_PP_CENTERS);
	int empty=0;
	int nonempty=0;
	for(OpenCVFeatureLoader::LocalFeatures::iterator iter=features.begin();
			iter!=features.end();iter++
			)
	{
		if((*iter).empty())
		{
			//clog<<"Empty descriptor matrix!"<<endl;
			empty++;
			continue;
		}
		nonempty++;
		trainer.add(*iter);
	}

	cout<<"Empty: "<<empty<<" Nonempty: "<<nonempty<<endl;
	Mat codebook = trainer.cluster();
	std::cout<<codebook;
	{
		DataSet<double> ds(outArg.getValue(), true,false,16 * 1024,
		            	true);
		ds.addMat(codebook,nameArg.getValue());
	}
	return 0;
}
