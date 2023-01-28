
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include "fc_m_resnet.hpp"
#include<vector>
#include<time.h>


//using namespace cv;
using namespace std;

vector<int> fisher_yates_shuffle(vector<int> table);
int main() {

 cout << "under work..." << endl;

  srand(time(NULL));

  const int tabl_size = 10;
  vector<int> table;
  table.clear();
  for(int i=0;i<tabl_size;i++)
  {
    table.push_back(0);
  }
  table = fisher_yates_shuffle(table);
  for(int i=0;i<tabl_size;i++)
  {
    cout << "rand table[" << i << "] = " << table[i] << endl;
  }

  fc_m_resnet fc_1;
  fc_1.get_version();
  


  //fc_1.randomize_weights(0.01);

  cv::Mat test;
  test.create(200, 400, CV_32FC1);
  float *index_ptr_testGrapics = test.ptr<float>(0);
  float pixel_level = 0.0f;
  for(int i=0; i< test.rows * test.cols;i++)
  {
    if(pixel_level < 1.0)
    {
      pixel_level = pixel_level + 0.00001f;
    }
    else{
      pixel_level = 0.0f;
    }
     *index_ptr_testGrapics = pixel_level;
    index_ptr_testGrapics++;
  }
  cv::imshow("diff", test);
  cv::waitKey(5000);
}

vector<int> fisher_yates_shuffle(vector<int> table) {
    int size = table.size();
    for (int i = 0; i < size; i++) {
        table[i] = i;
    }
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = table[i];
        table[i] = table[j];
        table[j] = temp;
    }
    return table;
}