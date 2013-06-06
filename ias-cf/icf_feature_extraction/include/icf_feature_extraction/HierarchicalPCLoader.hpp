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
#ifndef PCDLOADER_H
#define PCDLOADER_H

//#include "types.h"
#include <icf_dataset/DataSet.hpp>

//boost
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>

//stl
#include <map>
#include <fstream>
#include <algorithm>
#include <cfloat>

//ros for debuggong
#include<ros/ros.h>

using boost::shared_ptr;

using std::vector;
using std::string;
using boost::filesystem::is_directory;
using boost::filesystem::is_regular_file;
using boost::filesystem::path;
using boost::filesystem::directory_entry;
using boost::filesystem::directory_iterator;

using namespace cv;

namespace icf
{

bool hasEnding(std::string const &fullString, std::string const &ending)
{
  if (fullString.length() < ending.length())
    return false;
  size_t lastMatchPos = fullString.rfind(ending); // Find the last occurrence of ending
  bool isEnding = lastMatchPos != std::string::npos; // Make sure it's found at least once
  // If the string was found, make sure that any characters that follow it are the ones we're trying to ignore
  for (size_t i = lastMatchPos + ending.length(); (i < fullString.length()) && isEnding; i++)
  {
    if ((fullString[i] != '\n') && (fullString[i] != '\r'))
    {
      isEnding = false;
    }
  }
  return isEnding;
}

class PCDLoaderException : public std::exception
{
public:
  PCDLoaderException(const string& msg) throw () :
      msg(msg)
  {

  }
  ;

  virtual ~PCDLoaderException() throw ()
  {

  }
  ;

  virtual const char* what() const throw ()
  {
    return msg.c_str();
  }
  ;
private:
  string msg;
};

class FileLoader
{
public:
  typedef boost::shared_ptr<FileLoader> Ptr;
  virtual void loadFile(const path& path, const std::vector<int>& labels)=0;
  virtual void postProcessing()
  {
  }
  ;

  double getMsPerInputPointFeatureEstimationOnly()
  {
    return featureEstimationTiming / nrPointsProcessed;
  }

protected:
  double featureEstimationTiming;
  long nrPointsProcessed;
};

class SkipLoader : public FileLoader
{
public:
  virtual void loadFile(const path& path, const std::vector<int>& labels)
  {

  }

  virtual void postProcessing()
  {

  }
  ;
};

class LabelLoader : public FileLoader
{
private:
  std::string fileEnding;
public:

  LabelLoader(int skip = 5, std::string fileEnding = ".pcd") :
      fileEnding(fileEnding), skip(skip), skipcount(1)
  {

  }

  void loadFile(const path& p, const std::vector<int>& labels)
  {

    if (hasEnding(p.string(), fileEnding))
    {
      if (skipcount % skip != 0)
      {
        skipcount++;
        return;
      }
      skipcount = 1;
      this->labels.push_back(labels);
    }
  }

  vector<vector<int> > getLabels()
  {
    return this->labels;
  }

private:
  vector<vector<int> > labels;
  int skip;
  int skipcount;
};

class LabelNameLoader : public FileLoader
{
public:
  void loadFile(const path& p, const std::vector<int>& labels)
  {
    if (p.extension() == ".pcd")
    {
      this->labels.push_back(p.string());
    }
  }

  vector<string> getLabels()
  {
    return this->labels;
  }

private:
  vector<string> labels;
};

bool stringCompare(const string &left, const string &right)
{
  for (string::const_iterator lit = left.begin(), rit = right.begin(); lit != left.end() && rit != right.end();
      ++lit, ++rit)
    if (tolower(*lit) < tolower(*rit))
      return true;
    else if (tolower(*lit) > tolower(*rit))
      return false;
  if (left.size() < right.size())
    return true;
  return false;
}

bool cmpPath(path p1, path p2)
{
  return stringCompare(p1.string(), p2.string());
}

class HierarchicalPCDLoader
{
public:

  HierarchicalPCDLoader(FileLoader::Ptr fl, bool folderLabel = false) :
      fl(fl), folderLabel(folderLabel)
  {

  }

  virtual ~HierarchicalPCDLoader()
  {

  }

  void load(const path& p)
  {
    loadInternal(p, vector<int>());
    fl->postProcessing();
  }

  void loadInternal(const path& p, vector<int> labels)
  {
    if (exists(p))
    {
      if (is_directory(p))
      {

        directory_iterator iter(p);
        directory_iterator end;

        if (folderLabel)
        {
          labels.push_back(atoi(p.filename().c_str()));
        }
        else
        {
          labels.push_back(0);
        }
        std::vector<path> entries;
        std::vector<path> dirs;
        for (; iter != end; iter++)
        {
          directory_entry entry = *iter;
          const path& entryPath = entry.path();
          if (is_regular_file(entryPath))
          {
            int subClassLabel = labels.back();
            labels.pop_back();
            entries.push_back(entryPath);
            labels.push_back(subClassLabel);
          }
          else
          {
            std::cout << entryPath.string() << std::endl;
            dirs.push_back(entryPath);

          }
        }

        if (!entries.empty())
        {
          int subClassLabel = labels.back();
          labels.pop_back();
          sort(entries.begin(), entries.end(), cmpPath);
          for (std::vector<path>::iterator iter = entries.begin(); iter != entries.end(); iter++)
          {
            std::cout << "Processing file: " << (*iter).string() << std::endl;
            loadFile(*iter, labels);
          }
          labels.push_back(subClassLabel);
        }

        if (!dirs.empty())
        {
          sort(dirs.begin(), dirs.end(), cmpPath);
          for (std::vector<path>::iterator iter = dirs.begin(); iter != dirs.end(); iter++)
          {
            if (folderLabel)
            {
              labels.pop_back();
              labels.push_back(atoi((*iter).filename().c_str()));
            }
            else
            {
              labels.back() += 1;
            }
            std::cout << "Processing directory: " << (*iter).string() << std::endl;
            loadInternal(*iter, labels);
          }
        }
      }
    }
    else
    {
      //throw std::string("Root path doesn't exist ")+p.string();
    }
  }

  void loadFile(const path& base, const vector<int>& labels)
  {
    //Parallelize
    fl->loadFile(base, labels);
  }

private:
  FileLoader::Ptr fl;
  bool folderLabel;

};

}
;

#endif
