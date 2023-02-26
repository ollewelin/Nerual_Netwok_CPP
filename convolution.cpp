#include "convolution.hpp"
#include <iostream>
#include <fstream>
#include <math.h> 

using namespace std;

convolution::convolution()
{
    version_major = 0;
    version_mid = 0;
    version_minor = 1;
    //0.0.1 Almost empty class no algorithm yet implemented
    setup_state = 0;
    kernel_size = 3;
    stride = 0;
    cout << "Constructor Convloution neural network object " << endl;
}

convolution::~convolution()
{
    cout << "Destructor Convloution neural network object " << endl;
}

void convolution::set_kernel_size(int k_size)
{
    if(setup_state > 1)
    {
        cout << "Error could not change kernel when set_out_tensor_channels() already setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }

    if(k_size < 1)
    {
        cout << "Error kernel size is set < 1, k_size = " << k_size << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }

    int check_odd = k_size % 2;
    if(check_odd == 0)
    {
        cout << "Error kernel size is even. kernel size must always be an odd number. k_size = " << k_size << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }
    else
    {
        cout << "   OK kernel size = " << k_size << endl; 
    }
    input_side_size = input_tensor[0][0].size();
    if(input_side_size < k_size)
    {
        cout << "Error kernel size is > input side of square size input_side_size = " << input_side_size << endl; 
        cout << "kernel_size = " << k_size << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }
    else
    {
        cout << "   OK input_side_size = " << input_side_size << endl;
    }
    kernel_size = k_size;
    setup_state = 1;
}
int convolution::get_kernel_size()
{

    return kernel_size;
}
void convolution::set_stride(int stride_size)
{
    if(setup_state > 1)
    {
        cout << "Error could not change stride size when set_out_tensor_channels() already setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }
    if(stride_size > 1 || stride_size < 0)
    {
        cout << "   Error stride size out of range. stride = " << stride_size << endl; 
        cout << "   Stride size on convolution version " << ver_major << "." << version_mid << "." << version_minor << " is only support range 0-1" << endl;
        cout << "   Exit program" << endl; 
        exit(0);
    }
    cout << "   OK stride size is set to " << stride_size << endl; 
    stride = stride_size;
}
int convolution::get_stride()
{
    return stride;
}

void convolution::set_in_tensor(int total_input_size_one_channel, int in_channels)
{
    input_tensor_channels = in_channels;
    
    if(setup_state != 0)
    {
        cout << "Error could not set_in_tensor() setup_state must be = 0 when call set_in_tensor, setup_state = " << setup_state << endl; 
        cout << "setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }
    cout << "   data_size_one_sample one channel = " << total_input_size_one_channel << endl;
    root_of_intdata_size = sqrt(total_input_size_one_channel);
    cout << "   root_of_intdata_size = " << root_of_intdata_size << endl;
    int test_square_the_root = root_of_intdata_size * root_of_intdata_size;
    if (test_square_the_root != total_input_size_one_channel)
    {
        cout << "Error Indata one sample not able to make a square root without decimal, test_square_the_root = " << test_square_the_root << endl;
        cout << "Indata don't fit a perfect square, Exit program " << endl;
        exit(0);
    }
     
    vector<double> dummy_conv_1D_vector;
    vector<vector<double>> dummy_conv_2D_vector;
    //========= Set up convolution input tensor size for convolution object =================
    for(int i=0;i<root_of_intdata_size;i++)
    {
        dummy_conv_1D_vector.push_back(0.0);
    }
    for(int i=0;i<root_of_intdata_size;i++)
    {
        dummy_conv_2D_vector.push_back(dummy_conv_1D_vector);
    }
    for(int i=0;i<input_tensor_channels;i++)
    {
        input_tensor.push_back(dummy_conv_2D_vector);
        i_tensor_delta.push_back(dummy_conv_2D_vector);
    }
}

void convolution::set_out_tensor(int out_channels)
{
    output_tensor_channels = out_channels;
    if(setup_state != 1)
    {
        cout << "Error could not set_out_tensor() setup_state must be = 1 when call set_out_tensor_channels(), setup_state = " << setup_state << endl; 
        cout << "setup_state = " << setup_state << endl; 
        cout << "Exit program" << endl; 
        exit(0);
    }

    //========= Set up convolution weight tensor and output tensor size for convolution object =================
    if(stride == 0)
    {
        output_side_size = (input_side_size - kernel_size) + 1;
        
    }
    else if(stride == 1)
    {
        output_side_size = ((input_side_size - kernel_size) / 2) + 1;
    }
    cout << "   OK output_side_size = " << output_side_size << endl; 
    //output_side_size = xxx                                                            
    vector<double>                  dummy_1D_vect;
    vector<vector<double>>          dummy_2D_vect;
    vector<vector<vector<double>>>  dummy_3D_vect;
    for(int i=0;i<kernel_size;i++)
    {
        dummy_1D_vect.push_back(0.0);
    }
    for(int i=0;i<kernel_size;i++)
    {
        dummy_2D_vect.push_back(dummy_1D_vect);
    }
    for(int i=0;i<input_tensor_channels;i++)
    {
        dummy_3D_vect.push_back(dummy_2D_vect);
    }
    
    for(int i=0;i<output_tensor_channels;i++)
    {
        kernel_weights.push_back(dummy_3D_vect);//4D [output_channel][input_channel][kernel_row][kernel_col]
        change_weights.push_back(dummy_3D_vect);//4D [output_channel][input_channel][kernel_row][kernel_col]
        kernel_deltas.push_back(dummy_3D_vect);//4D [output_channel][input_channel][kernel_row][kernel_col]
    }
    
    //Add also one bias weight at end for the hole [input_channel][0][0]
    for(int i=0;i<output_tensor_channels;i++)
    {
        kernel_bias_weights.push_back(0.0);//kernel bias weight [output_channel]
        change_bias_weights.push_back(0.0);//change bias weight [output_channel]
        //kernel_deltas dont need space for bias. no bias here
    }
    cout << "   kernel_bias_weights.size() = " << kernel_bias_weights.size() << endl;
    cout << "   kernel_weights.size() = " << kernel_weights.size() << endl;
    cout << "   kernel_weights[0].size() = " << kernel_weights[0].size() << endl;
    cout << "   kernel_weights[0][0].size() = " << kernel_weights[0][0].size() << endl;
    cout << "   kernel_weights[" << output_tensor_channels - 1 << "][" << input_tensor_channels - 1 << "][" << kernel_size - 1 << "].size() = " << kernel_weights[output_tensor_channels - 1][input_tensor_channels - 1][kernel_size - 1].size() << endl;
    cout << "   input_tensor.size() = " << input_tensor.size() << endl;
    cout << "   output_tensor.size() = " << output_tensor.size() << endl;

    cout << endl;

    dummy_1D_vect.clear();
    dummy_2D_vect.clear();
    for(int i=0;i<output_side_size;i++)
    {
        dummy_1D_vect.push_back(0.0);
    }
    for(int i=0;i<output_side_size;i++)
    {
        dummy_2D_vect.push_back(dummy_1D_vect);
    }
    for(int i=0;i<output_tensor_channels;i++)
    {
        output_tensor.push_back(dummy_2D_vect);
        o_tensor_delta.push_back(dummy_2D_vect);
    }
    setup_state = 2;
}
void convolution::randomize_weights(double rand_prop)
{
    
}
void convolution::load_weights(string filename)
{
    
}
void convolution::save_weights(string filename)
{
    
}
void convolution::conv_forward()
{
    
}
void convolution::conv_backprop()
{
    
}
void convolution::conv_update_weights()
{
    
}
void convolution::conv_transpose_fwd()
{
    
}
void convolution::conv_transpose_bkp()
{
    
}

void convolution::get_version()
{
    cout << "Convloution neural network object" << endl;
    cout << "Convloution version : " << version_major << "." << version_mid << "." << version_minor << endl;
    ver_major = version_major;
    ver_mid = version_mid;
    ver_minor = version_minor;
}