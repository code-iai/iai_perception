/*
 * Copyright (c) 2012,
 * Zoltan-Csaba Marton <marton@cs.tum.edu>,
 * Ferenc Balint-Benczedi <balintb.ferenc@gmail.com>,
 * Florian Seidel <seidel.florian@gmail.com
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
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <tclap/CmdLine.h>
#include <icf_dataset/DataSet.hpp>

using namespace icf;
using namespace TCLAP;

int main(int argc, char ** argv)
{
  //Command line parser
  CmdLine cmdLine("join_datasets", ' ', "dev");
  ValueArg<std::string> aArg("a", "file1", "name of file 1", true, "a", "string");
  ValueArg<std::string> bArg("b", "file2", "name of file 2", true, "b", "string"); // TODO: create if not exisiting
  ValueArg<std::string> sArg("s", "source", "name of field form file 1 to be copied into file 2 (default: /y)", false, "/y",
                        "string");
  ValueArg<std::string> tArg("t", "target", "new name of copied field in file 2 (default: /y)", false, "/y", "string");
  cmdLine.add(tArg);
  cmdLine.add(sArg);
  cmdLine.add(bArg);
  cmdLine.add(aArg);
  cmdLine.parse(argc, argv);

  try
  {
    DS ds1(aArg.getValue(), false, true);
    DS ds2(bArg.getValue());
    DS::MatrixPtr m;
    if (ds1.contains(sArg.getValue()))
      m = ds1.getFeatureMatrix(sArg.getValue());
    else
    {
      ROS_ERROR(
          "Argument are not correct! Dataset %s does not contain any field named %s", aArg.getValue().c_str(), sArg.getValue().c_str());
      return -1;
    }
    if (!ds2.contains(tArg.getValue()))
    {
      ds2.setFeatureMatrix(m, tArg.getValue());
    }
    else
    {
      ROS_ERROR("Dataset %s allready has a field named %s", bArg.getValue().c_str(), tArg.getValue().c_str());
      return -1;
    }
  }
  catch (ICFException& e)
  {
    std::cerr << boost::diagnostic_information(e);
    return -1;
  }
  return 0;
}
