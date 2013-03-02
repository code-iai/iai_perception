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

#ifndef IMLOADERS_HPP_
#define IMLOADERS_HPP_

#include <icf_feature_extraction/HierarchicalPCLoader.hpp>
#include <opencv2/opencv.hpp>

namespace icf{

class ImageLoader : public FileLoader {
public:
	ImageLoader(string img_ext, string mask,int skip=1):
			img_ext(img_ext),mask_ext(mask),skip(skip),skipped(0)
	{

	}

	virtual void loadFile(const path& p, const std::vector<int>& labels)
	{
		if(hasEnding(p.string(),img_ext))
		{
			if(skipped%skip!=0)
			{
				std::cout<<"Skipped "<<p.string()<<std::endl;
				skipped++;
				return;
			}
			skipped=1;
			Mat img = imread(p.string());
			Mat mask= imread(p.string().substr(0,p.string().size()-	img_ext.size())+mask_ext,0);
			if(img.empty())
			{
				std::cerr<<"Could not load image file: "<<p.string()<<std::endl;
				//exit(-1);
				return;
			}
			if(mask.empty())
			{
				std::cerr<<"Could not load mask file: "<<p.string().substr(0,p.string().size()-	img_ext.size())+mask_ext<<std::endl;
				//exit(-1);
				return;
			}
			processImage(labels,img,mask);
		}
	}

	virtual void processImage(const std::vector<int>& labels,Mat& img, Mat& mask)=0;

protected:
	string img_ext;
	string mask_ext;

	bool hasEnding (string const &fullString, string const &ending)
	{
		if (fullString.length() > ending.length()) {
			return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
		} else {
			return false;
		}
	}
private:
	int skip;
	int skipped;
};

class SHHistogramLoader : public ImageLoader {
private:
	int sbins;
	int hbins;
	Mat histograms;
public:

	SHHistogramLoader(string imgExt, string maskExt, int skip, int sbins,int hbins):
		ImageLoader(imgExt,maskExt,skip),sbins(sbins),hbins(hbins)
	{

	}

	Mat getHistograms()
	{
		return histograms;
	}

	virtual void processImage(const std::vector<int>& labels, Mat& img, Mat& mask)
	{
	    Mat hsv;
	    cvtColor(img, hsv, CV_RGB2HSV);

	    int histSize[] = {hbins, sbins};
	    // hue varies from 0 to 179, see cvtColor
	    float hranges[] = { 0, 180 };
	    // saturation varies from 0 (black-gray-white) to
	    // 255 (pure spectrum color)
	    float sranges[] = { 0, 256 };
	    const float* ranges[] = { hranges, sranges };
	    Mat hist;
	    // we compute the histogram from the 0-th and 1-st channels
	    int channels[] = {0, 1};

	    calcHist( &hsv, 1, channels, mask, // do not use mask
	        hist, 2, histSize, ranges,
	        true, // the histogram is uniform
	        false );
	    //std::cout<<hist.rows<<" "<<hist.cols<<std::endl;
	    //std::cout<<hist.reshape(0,1)<<std::endl;
	    histograms.push_back(hist.reshape(0,1));
	}


};

class OpenCVFeatureLoader :public ImageLoader {
public:

	typedef vector<Mat> LocalFeatures;

	OpenCVFeatureLoader(string img_ext, string mask,string featureDetector, string descriptorExtractor,int skip=1)
		:ImageLoader(img_ext,mask,skip)
	{
		detector = FeatureDetector::create(featureDetector);
		extractor = DescriptorExtractor::create(descriptorExtractor);
	}

	virtual void processImage(const std::vector<int>& labels, Mat& img, Mat& mask)
	{
		vector<KeyPoint> keypoints;
		detector->detect(img,keypoints,mask);
		Mat descriptors;
		extractor->compute(img,keypoints,descriptors);
		std::cout<<descriptors;
		features.push_back(descriptors);
	}

	LocalFeatures getLocalFeatures()
	{
		return this->features;
	}

	virtual void postProcessing() {

		}
		;

private:

	LocalFeatures features;
	cv::Ptr<FeatureDetector> detector;
	cv::Ptr<DescriptorExtractor> extractor;
	string img_ext;
	string mask;

};

class OpenCVBOWLoader : public ImageLoader {

public:

	OpenCVBOWLoader(string img_ext, string mask, string featureDetector, string descriptorExtractor,
			string descriptorMatcher,
			Mat& codebook, int skip=1
			):ImageLoader(img_ext,mask,skip),row(0),codebook(codebook),m(DataSet<double>::Matrix(1,codebook.rows))
	{
		matcher=DescriptorMatcher::create(descriptorMatcher);
		extractor=DescriptorExtractor::create(descriptorExtractor);
		detector=FeatureDetector::create(featureDetector);
		bowDE=new BOWImgDescriptorExtractor(extractor,matcher);
		bowDE->setVocabulary(this->codebook);
	}

	virtual ~OpenCVBOWLoader()
	{
		delete bowDE;
	}

	virtual void processImage(const std::vector<int>& labels, Mat& img, Mat& mask)
	{

		vector<KeyPoint> keypoints;
		if(!mask.empty())
		{
			detector->detect(img,keypoints,mask);
		}
		else
			detector->detect(img,keypoints);

		Mat descriptor;
		bowDE->compute(img,keypoints,descriptor,NULL,NULL);

		if(row>=m.rows())
			increaseSize();

		if(descriptor.empty())
		{
			std::cerr<<"No keypoints detected"<<std::endl;
		    descriptor = Mat( 1, this->codebook.rows, bowDE->descriptorType(), Scalar::all(0.0) );
		}
		//cout<<descriptor<<std::endl;
	    float *dptr = descriptor.ptr<float>(0);
		for(int i=0;i<descriptor.cols;i++)
		{
			m(row,i)=dptr[i];
		}
		row++;
	}

	DataSet<double>::Matrix getMatrix()
	{
		return this->m;
	}

	virtual void postProcessing()
	{
		pruneSize();
	}

private:
	void pruneSize() {
		DataSet<double>::Matrix newm(row, m.cols());
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < m.cols(); j++) {
				newm(i, j) = m(i, j);
			}
		}
		m = newm;
	}
	void increaseSize() {
		DataSet<double>::Matrix newm(m.rows() * 2, m.cols());
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.cols(); j++) {
				newm(i, j) = m(i, j);
			}
		}
		m = newm;

	}

	int row;
	Mat& codebook;
	DataSet<double>::Matrix m;
	cv::Ptr<DescriptorMatcher> matcher;
	cv::Ptr<DescriptorExtractor> extractor;
	cv::Ptr<FeatureDetector> detector;
	BOWImgDescriptorExtractor* bowDE;
};
}
#endif /* IMLOADERS_HPP_ */
