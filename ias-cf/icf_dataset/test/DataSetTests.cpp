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

#include <icf_dataset/DataSet.hpp>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

using namespace icf;
using namespace std;
using namespace cv;

TEST(DataSetTest,loadFromFileAndSerializeDeserialize)
{
  hsize_t size = 0;
  uint8_t *buffer;
  {
    DS ds("data/test1.hdf5", false, false);
    DS::MatrixPtr data_ptr = ds.getFeatureMatrix("/test_matrix");
    EXPECT_TRUE(data_ptr->rows()==5);
    EXPECT_TRUE(data_ptr->cols()==5);
    buffer = ds.serialize(size);
    EXPECT_TRUE(size!=0);
  }

  {
    DS ds("data/deserialization_test23333.hdf5", buffer, size, false);
    DS::MatrixPtr m_ptr = ds.getFeatureMatrix("/test_matrix");
    for (int i = 0; i < m_ptr->rows(); i++)
    {
      for (int j = 0; j < m_ptr->cols(); j++)
      {
        //std::cout<<(*m_ptr)(i,j)<<std::endl;
        EXPECT_TRUE((*m_ptr)(i,j)==i*j || (*m_ptr)(i,j)==1234 || (i==3 && j==3));
      }
    }
  }

}

TEST(DataSetTest,testContains)
{
  DS ds;
  DS::Matrix m(5, 5);
  for (int i = 0; i < m.rows(); i++)
  {
    for (int j = 0; j < m.cols(); j++)
    {
      m(i, j) = i * j;
    }
  }
  ds.setFeatureMatrix(m, "/test");
  EXPECT_TRUE(ds.contains("/test"));
  EXPECT_FALSE(ds.contains("/not_contained"));

}

TEST(DataSetTest,testAlias)
{
  DS ds;
  DS::Matrix m(5, 5);
  for (int i = 0; i < m.rows(); i++)
  {
    for (int j = 0; j < m.cols(); j++)
    {
      m(i, j) = i * j;
    }
  }
  ds.setFeatureMatrix(m, "/test");
  ds.alias("/test", "/new_name");
  EXPECT_TRUE(ds.contains("/test"));
  EXPECT_TRUE(ds.contains("/new_name"));

  DS::Matrix m2(5, 5);
  for (int i = 0; i < m.rows(); i++)
  {
    for (int j = 0; j < m.cols(); j++)
    {
      m2(i, j) = 3333;
    }
  }
  ds.setFeatureMatrix(m2, "/test2");
  ds.alias("/test2", "/new_name");
  EXPECT_TRUE(ds.getFeatureMatrix("/new_name")->operator ()(1,1)==3333);
}

TEST(DataSetTest,testRename)
{
  DS ds;
  DS::Matrix m(5, 5);
  for (int i = 0; i < m.rows(); i++)
  {
    for (int j = 0; j < m.cols(); j++)
    {
      m(i, j) = i * j;
    }
  }
  ds.setFeatureMatrix(m, "/test");
  ds.renameFeatureMatrix("/test", "/new_name");
  EXPECT_FALSE(ds.contains("/test"));
  EXPECT_TRUE(ds.contains("/new_name"));
}

TEST(DataSetTest,testSerializationToStreamAndBack)
{

  std::stringstream ss;
  {
    DS ds("data/serialization_test3313.hdf5", true, true, 1024, false);
    DS::Matrix m(5, 5);
    for (int i = 0; i < m.rows(); i++)
    {
      for (int j = 0; j < m.cols(); j++)
      {
        m(i, j) = i * j;
      }
    }
    ds.setFeatureMatrix(m, "/test");
    ss << ds;

  }

  {
    DS ds("data/deserialization_test123.hdf5", (void*)ss.str().c_str(), ss.str().length(), false);
    DS::MatrixPtr m_ptr = ds.getFeatureMatrix("/test");
    for (int i = 0; i < m_ptr->rows(); i++)
    {
      for (int j = 0; j < m_ptr->cols(); j++)
      {
        EXPECT_TRUE((*m_ptr)(i,j)==i*j);
      }
    }
  }

}

TEST(DataSetTest, testSerialization)
{
  uint8_t * buffer;
  hsize_t size = 0;
  {
    DS ds("data/serialization_test.hdf5", true, true, 1024, false);
    DS::Matrix m(5, 5);
    for (int i = 0; i < m.rows(); i++)
    {
      for (int j = 0; j < m.cols(); j++)
      {
        m(i, j) = i * j;
      }
    }
    ds.setFeatureMatrix(m, "/test");
    buffer = ds.serialize(size);
    EXPECT_TRUE(size!=0);
  }

  {
    DS ds("data/deserialization_test41266.hdf5", buffer, size, false);
    DS::MatrixPtr m_ptr = ds.getFeatureMatrix("/test");
    for (int i = 0; i < m_ptr->rows(); i++)
    {
      for (int j = 0; j < m_ptr->cols(); j++)
      {
        EXPECT_TRUE((*m_ptr)(i,j)==i*j);
      }
    }
  }

}

TEST(DataSetTest, testOpenCVFunctionality)
{
  {
    DS ds("data/test_opencv.hdf5", true, true, 1024, false);
    Mat_<double> mat(5, 5);
    for (int i = 0; i < mat.rows; i++)
    {
      for (int j = 0; j < mat.cols; j++)
      {
        mat(i, j) = i * j;
      }
    }
    ds.addMat(mat, "/ocmat");

    DS::MatrixPtr m = ds.getFeatureMatrix("/ocmat");
    for (int i = 0; i < m->rows(); i++)
    {
      for (int j = 0; j < m->cols(); j++)
      {
        EXPECT_TRUE((*m)(i,j)==i*j);
      }
    }
  }
}

TEST(DataSetTest, testPersistance)
{
  {
    DS ds("data/test1.hdf5", true, true, 1024, true);
    DS::Matrix m(5, 5);
    for (int i = 0; i < m.rows(); i++)
    {
      for (int j = 0; j < m.cols(); j++)
      {
        m(i, j) = i * j;
      }
    }
    ds.setFeatureMatrix(m, "/test_matrix");
  } //should have written to disk now

  //read back in check content, modify, write back out.
  {
    DS ds("data/test1.hdf5", false, true, 1024, true);
    DS::MatrixPtr m_ptr = ds.getFeatureMatrix("/test_matrix");
    for (int i = 0; i < m_ptr->rows(); i++)
    {
      for (int j = 0; j < m_ptr->cols(); j++)
      {
        EXPECT_TRUE((*m_ptr)(i,j)==i*j);
      }
    }
    (*m_ptr)(2, 2) = 1234;
    ds.setFeatureMatrix(m_ptr, "/test_matrix");
  }

  //check if modifications are on disk, modify some more without writing back
  {
    DS ds("data/test1.hdf5", false, true, 1024, false);
    DS::MatrixPtr m_ptr = ds.getFeatureMatrix("/test_matrix");
    EXPECT_TRUE((*m_ptr)(2,2)==1234);

    (*m_ptr)(3, 3) = 0;
    ds.setFeatureMatrix(m_ptr, "/test_matrix");
  }

  //check if stuff was written back
  {
    DS ds("data/test1.hdf5", false, true, 1024, false);
    DS::MatrixPtr m_ptr = ds.getFeatureMatrix("/test_matrix");
    EXPECT_TRUE((*m_ptr)(3,3)!=0);
  }

  //check flush option
  {
    DS ds("data/test2.hdf5", true, true, 1024, false);
    double a[4] = {1, 2, 3, 4};
    ds.setFeatureMatrix(a, 2, 2, "/test");
  }

  // this causes an exception to be raised if flush option works correctly
  {
    bool threw = false;
    try
    {
      DS ds("data/test2.hdf5", false, true, 1014, false);
    }
    catch (ICFException& e)
    {
      threw = true;
    }
    EXPECT_TRUE(threw);
  }
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
