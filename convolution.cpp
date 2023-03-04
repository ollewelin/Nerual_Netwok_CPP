#include "convolution.hpp"
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

convolution::convolution()
{
    version_major = 0;
    version_mid = 0;
    version_minor = 0;
    // 0.0.0 Not finnish at all
    setup_state = 0;
    kernel_size = 3;
    stride = 1;
    dropout_proportion = 0.0;
    use_dopouts = 0;
    cout << "Constructor Convloution neural network object " << endl;
}

convolution::~convolution()
{
    cout << "Destructor Convloution neural network object " << endl;
}
void convolution::set_kernel_size(int k_size)
{
    if (setup_state > 0)
    {
        cout << "Error could not change kernel already set up once setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    if (k_size < 1)
    {
        cout << "Error kernel size is set < 1, k_size = " << k_size << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    int check_odd = k_size % 2;
    if (check_odd == 0)
    {
        cout << "Error kernel size is even. kernel size must always be an odd number. k_size = " << k_size << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    else
    {
        //  cout << "   OK kernel size = " << k_size << endl;
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
    if (setup_state > 1)
    {
        cout << "Error must set stride after kernel setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    if (stride_size < 1)
    {
        cout << "   Error stride size out of range. stride = " << stride_size << endl;
        cout << "   Stride size on convolution version " << ver_major << "." << version_mid << "." << version_minor << " is only support range 1-n" << endl;
        cout << "   Exit program" << endl;
        exit(0);
    }
    cout << "   ========================================" << endl;
    cout << "   OK stride size is set to " << stride_size << endl;
    stride = stride_size;
    setup_state = 2;
}
int convolution::get_stride()
{
    return stride;
}

void convolution::set_in_tensor(int total_input_size_one_channel, int in_channels)
{
    input_tensor_channels = in_channels;

    if (setup_state != 2)
    {
        cout << "Error could not set_in_tensor() setup_state must be = 2 when call set_in_tensor, setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    cout << "   data_size_one_sample one channel = " << total_input_size_one_channel << endl;
    root_of_intdata_size = sqrt(total_input_size_one_channel);
    // cout << "   root_of_intdata_size = " << root_of_intdata_size << endl;
    int test_square_the_root = root_of_intdata_size * root_of_intdata_size;
    if (test_square_the_root != total_input_size_one_channel)
    {
        cout << "Error Indata one sample not able to make a square root without decimal, test_square_the_root = " << test_square_the_root << endl;
        cout << "Indata don't fit a perfect square, Exit program " << endl;
        exit(0);
    }

    vector<double> dummy_conv_1D_vector;
    vector<vector<double>> dummy_conv_2D_vector;

    // Add stride add calculation at right and bottom side if neccesary
    int add_side = 0;
    if (stride > 1)
    {
        cout << "   Start doing calculatiote if stride add to input_tensor_size is neccesary" << endl;
        int modulo = (root_of_intdata_size - kernel_size) % stride; // Calculation if add
        if (modulo > 0)
        {
            add_side = stride - modulo;
        }
        cout << "   root_of_intdata_size = " << root_of_intdata_size << endl;
        cout << "   kernel_size = " << kernel_size << endl;
        cout << "   stride = " << stride << endl;
        cout << "   add_side = " << add_side << endl;
        if (add_side > 0)
        {
            cout << "   Note! Add bottom rows and right column at input_tensor to make the convolution betwhen input and kernel not miss any input data during stride stop operation " << endl;
        }
    }
    int temporary_input_t_side_size = root_of_intdata_size + add_side;
    //========= Set up convolution input tensor size for convolution object =================
    for (int i = 0; i < temporary_input_t_side_size; i++)
    {
        dummy_conv_1D_vector.push_back(0.0);
    }
    for (int i = 0; i < temporary_input_t_side_size; i++)
    {
        dummy_conv_2D_vector.push_back(dummy_conv_1D_vector);
    }
    for (int i = 0; i < input_tensor_channels; i++)
    {
        input_tensor.push_back(dummy_conv_2D_vector);
        i_tensor_delta.push_back(dummy_conv_2D_vector);
    }
    input_side_size = input_tensor[0][0].size();
    if (input_side_size < kernel_size)
    {
        cout << "Error kernel size is > input side of square size input_side_size = " << input_side_size << endl;
        cout << "kernel_size = " << kernel_size << endl;
        cout << "Exit program" << endl;
        exit(0);
    }
    else
    {
        cout << "   OK input_side_size = " << input_side_size << endl;
    }

    setup_state = 3;
}

void convolution::set_out_tensor(int out_channels)
{
    output_tensor_channels = out_channels;
    if (setup_state != 3)
    {
        cout << "Error could not set_out_tensor() setup_state must be = 3 when call set_out_tensor_channels(), setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    //========= Set up convolution weight tensor and output tensor size for convolution object =================
    output_side_size = ((input_side_size - kernel_size) / stride) + 1;
    if (output_side_size > 0)
    {
        cout << "   OK output_side_size = " << output_side_size << endl;
    }
    else
    {
        cout << "Error efter stride and kernel calculation for output size. output_side_size = " << output_side_size << endl;
        cout << "decrease stide of layers or increase input size" << endl;
        cout << "Exit program" << endl;
        exit(0);
    }

    // output_side_size = xxx
    vector<double> dummy_1D_vect;
    vector<vector<double>> dummy_2D_vect;
    vector<vector<vector<double>>> dummy_3D_vect;
    for (int i = 0; i < kernel_size; i++)
    {
        dummy_1D_vect.push_back(0.0);
    }
    for (int i = 0; i < kernel_size; i++)
    {
        dummy_2D_vect.push_back(dummy_1D_vect);
    }
    for (int i = 0; i < input_tensor_channels; i++)
    {
        dummy_3D_vect.push_back(dummy_2D_vect);
    }

    for (int i = 0; i < output_tensor_channels; i++)
    {
        kernel_weights.push_back(dummy_3D_vect); // 4D [output_channel][input_channel][kernel_row][kernel_col]
        change_weights.push_back(dummy_3D_vect); // 4D [output_channel][input_channel][kernel_row][kernel_col]
        kernel_deltas.push_back(dummy_3D_vect);  // 4D [output_channel][input_channel][kernel_row][kernel_col]
    }

    // Add also one bias weight at end for the hole [input_channel][0][0]
    for (int i = 0; i < output_tensor_channels; i++)
    {
        accum_bias_deltas.push_back(0.0);   //
        kernel_bias_weights.push_back(0.0); // kernel bias weight [output_channel]
        change_bias_weights.push_back(0.0); // change bias weight [output_channel]
        // kernel_deltas dont need space for bias. no bias here
    }

    dummy_1D_vect.clear();
    dummy_2D_vect.clear();
    for (int i = 0; i < output_side_size; i++)
    {
        dummy_1D_vect.push_back(0.0);
    }
    for (int i = 0; i < output_side_size; i++)
    {
        dummy_2D_vect.push_back(dummy_1D_vect);
    }
    for (int i = 0; i < output_tensor_channels; i++)
    {
        output_tensor.push_back(dummy_2D_vect);
        o_tensor_delta.push_back(dummy_2D_vect);
        internal_tensor_delta.push_back(dummy_2D_vect); // o_tensor_delta derivated backwards to inside the activation fucntion
    }

    cout << "   kernel_bias_weights.size() = " << kernel_bias_weights.size() << endl;
    cout << "   kernel_weights.size() = " << kernel_weights.size() << endl;
    cout << "   kernel_weights[0].size() = " << kernel_weights[0].size() << endl;
    cout << "   kernel_weights[0][0].size() = " << kernel_weights[0][0].size() << endl;
    cout << "   kernel_weights[" << output_tensor_channels - 1 << "][" << input_tensor_channels - 1 << "][" << kernel_size - 1 << "].size() = " << kernel_weights[output_tensor_channels - 1][input_tensor_channels - 1][kernel_size - 1].size() << endl;
    cout << "   input_tensor.size() = " << input_tensor.size() << endl;
    cout << "   output_tensor.size() = " << output_tensor.size() << endl;
    cout << "   output_tensor[" << output_tensor_channels - 1 << "][" << output_side_size - 1 << "].size() = " << output_tensor[output_tensor_channels - 1][output_side_size - 1].size() << endl;
    cout << "   ========================================" << endl;

    setup_state = 4;
}
void convolution::randomize_weights(double rand_prop)
{
    if (setup_state != 4)
    {
        cout << "Error could not radmoize_weight() setup_state must be = 4, setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        // TODO
        exit(0);
    }
    setup_state = 5;
}
void convolution::load_weights(string filename)
{
    if (setup_state != 4)
    {
        cout << "Error could not load_weights() setup_state must be = 4, setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        // TODO
        exit(0);
    }
    setup_state = 5;
}
void convolution::save_weights(string filename)
{
}

double convolution::activation_function(double input_data)
{
    double output_data = 0.0;
    int this_node_dopped_out = 0;
    if (use_dopouts == 1)
    {
        double dropout_random = ((double)rand() / RAND_MAX);
        if (dropout_random < dropout_proportion)
        {
            this_node_dopped_out = 1;
        }
    }
    if (this_node_dopped_out == 0)
    {
        if (activation_function_mode == 0)
        {
            // 0 = sigmoid activation function
            output_data = 1.0 / (1.0 + exp(-input_data)); // Sigmoid function and put it into
        }
        else
        {
            // 1 = Relu simple activation function
            // 2 = Relu fix leaky activation function
            // 3 = Relu random variable leaky activation function
            if (input_data >= 0.0)
            {
                // Positiv value go right though ar Relu (Rectify Linear)
                output_data = input_data;
                // cout << "forward + output_data = " << output_data << endl;
            }
            else
            {
                // Negative
                switch (activation_function_mode)
                {
                case 1:
                    // 1 = Relu simple activation function
                    output_data = 0.0;
                    //  cout << "forward - output_data = " << output_data << endl;
                    break;
                case 2:
                    // 2 = Relu fix leaky activation function
                    output_data = input_data * fix_leaky_proportion;
                    //  cout << "forward -2 output_data = " << output_data << endl;
                    break;
                }
            }
        }
    }
    return output_data;
}
double convolution::delta_activation_func(double delta_outside_function, double value_from_node_outputs)
{
    double delta_inside_func = 0.0;
    if (activation_function_mode == 0)
    {
        // 0 = sigmoid activation function
        delta_inside_func = delta_outside_function * value_from_node_outputs * (1.0 - value_from_node_outputs); // Sigmoid function and put it into
    }
    else
    {
        // 1 = Relu simple activation function
        // 2 = Relu fix leaky activation function
        // 3 = Relu random variable leaky activation function
        if (value_from_node_outputs >= 0.0)
        {
            // Positiv value go right though ar Relu (Rectify Linear)
            delta_inside_func = delta_outside_function;
        }
        else
        {
            // Negative
            switch (activation_function_mode)
            {
            case 1:
                // 1 = Relu simple activation function
                delta_inside_func = 0;
                break;
            case 2:
                // 2 = Relu fix leaky activation function

                delta_inside_func = delta_outside_function * fix_leaky_proportion;
                break;
            }
        }
    }
    if (value_from_node_outputs == 0.0)
    {
        delta_inside_func = 0.0; // Dropout may have ocure
    }
    return delta_inside_func;
}

// More computation intensive metode. But vetor memory pipline large jumping metode
void convolution::conv_forward1()
{
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        for (int y_slide = 0; y_slide < output_side_size; y_slide++)
        {
            for (int x_slide = 0; x_slide < output_side_size; x_slide++)
            {
                // Make the dot product of input tensor with kernel wheight for one kernel position
                double dot_product = kernel_bias_weights[out_ch_cnt]; // start with bias value for the dot product
                for (int ky = 0; ky < kernel_size; ky++)
                {
                    int inp_tens_y_pos = ky + y_slide * stride;
                    for (int kx = 0; kx < kernel_size; kx++)
                    {

                        int inp_tens_x_pos = kx + x_slide * stride;
                        // Itterate dot product
                        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                        {
                            dot_product += input_tensor[in_ch_cnt][inp_tens_y_pos][inp_tens_x_pos] * kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx];
                        }
                    }
                }
                // Make the activation function
                // Put the dot product to output tensor map
                output_tensor[out_ch_cnt][y_slide][x_slide] = activation_function(dot_product);
            }
        }
    }
}

// Less vetor memory large pipline jumping metode But more computation intensive metode

void convolution::conv_forward2()
{
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
        {
            for (int y_slide = 0; y_slide < output_side_size; y_slide++)
            {
                for (int x_slide = 0; x_slide < output_side_size; x_slide++)
                {
                    // Make the dot product of input tensor with kernel wheight for one kernel position
                    double dot_product = kernel_bias_weights[out_ch_cnt]; // start with bias value for the dot product
                    for (int ky = 0; ky < kernel_size; ky++)
                    {
                        int inp_tens_y_pos = ky + y_slide * stride;
                        for (int kx = 0; kx < kernel_size; kx++)
                        {

                            int inp_tens_x_pos = kx + x_slide * stride;
                            // Itterate dot product
                            dot_product += input_tensor[in_ch_cnt][inp_tens_y_pos][inp_tens_x_pos] * kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx];
                        }
                    }
                    // Make the activation function
                    // Put the dot product to output tensor map
                    output_tensor[out_ch_cnt][y_slide][x_slide] = activation_function(dot_product);
                }
            }
        }
    }
}

void convolution::xy_start_stop_kernel(int slide_val)
{
    start_ret = slide_val % stride;
    int start_constraint_end = (output_side_size * stride - kernel_size / 2);
    if (slide_val > start_constraint_end)
    {
        start_ret = start_ret + (slide_val - output_side_size * stride);
    }
    stop_ret = slide_val + 1;
    if (stop_ret > kernel_size)
    {
        stop_ret = kernel_size;
    }
    // Check and debug algorithm
    if (start_ret > kernel_size - 1)
    {
        cout << "debug algorithm start_ret = " << start_ret << endl;
        exit(0);
    }
}

void convolution::conv_backprop()
{
    // Compute delta for each output channel
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        accum_bias_deltas[out_ch_cnt] = 0.0; // used only for bias weight update later
        for (int y_slide = 0; y_slide < output_side_size; y_slide++)
        {
            for (int x_slide = 0; x_slide < output_side_size; x_slide++)
            {
                // Compute derivative of activation function
                double delta_activation = delta_activation_func(o_tensor_delta[out_ch_cnt][y_slide][x_slide], output_tensor[out_ch_cnt][y_slide][x_slide]);
                // delta_activation used in this loop for kernel weight delta calculations
                // store also delta_activation into internal_tensor_delta[][][] used later for delta for input tensor calculation
                internal_tensor_delta[out_ch_cnt][y_slide][x_slide] = delta_activation;
                // Update delta for bias weights
                accum_bias_deltas[out_ch_cnt] += delta_activation; // used only for bias weight update later
                                                                   // Update delta for kernel weights and input tensor
                for (int ky = 0; ky < kernel_size; ky++)
                {
                    int inp_tens_y_pos = ky + y_slide * stride;
                    for (int kx = 0; kx < kernel_size; kx++)
                    {
                        int inp_tens_x_pos = kx + x_slide * stride;
                        // Update delta for kernel weight
                        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                        {
                            kernel_deltas[out_ch_cnt][in_ch_cnt][ky][kx] += delta_activation * input_tensor[in_ch_cnt][inp_tens_y_pos][inp_tens_x_pos];
                        }
                    }
                }
            }
        }
    }

    // Update delta for input tensor. Flipped 180 deg kernel_weight
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        for (int y_slide = 0; y_slide < input_side_size; y_slide++)
        {
            xy_start_stop_kernel(y_slide);
            int y_start_ret = start_ret;
            int y_stop_ret = stop_ret;
            for (int x_slide = 0; x_slide < input_side_size; x_slide++)
            {
                xy_start_stop_kernel(x_slide);
                int x_start_ret = start_ret;
                int x_stop_ret = stop_ret;
                int yi = 0;
                for (int ky = y_start_ret; ky < y_stop_ret; ky = ky + stride) // Flipped 180 deg kernel_weight
                {
                    int out_tens_y_pos = (y_slide / stride) + yi; //
                    yi--;
                    if (out_tens_y_pos < 0)
                    {
                        out_tens_y_pos = 0;
                    }

                    if (out_tens_y_pos >= output_side_size - 1)
                    {
                        out_tens_y_pos = output_side_size - 1;
                    }
                    int xi = 0;
                    for (int kx = x_start_ret; kx < x_stop_ret; kx = kx + stride) // Flipped 180 deg kernel_weight
                    {
                        int out_tens_x_pos = (x_slide / stride) + xi;
                        xi--;
                        if (out_tens_x_pos < 0)
                        {
                            out_tens_x_pos = 0;
                        }
                        if (out_tens_x_pos >= output_side_size - 1)
                        {
                            out_tens_x_pos = output_side_size - 1;
                        }
                        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                        {
                            // Update delta for input tensor. Flipped 180 deg kernel_weight
                            i_tensor_delta[in_ch_cnt][y_slide][x_slide] += internal_tensor_delta[out_ch_cnt][out_tens_y_pos][out_tens_x_pos] * kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx];
                        }
                    }
                }
            }
        }
    }
}

void convolution::conv_update_weights()
{
    // Update kernel weights
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        change_bias_weights[out_ch_cnt] = (learning_rate * accum_bias_deltas[out_ch_cnt]) + (momentum * change_bias_weights[out_ch_cnt]);
        kernel_bias_weights[out_ch_cnt] += change_bias_weights[out_ch_cnt];
        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
        {
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    change_weights[out_ch_cnt][in_ch_cnt][ky][kx] = (learning_rate * kernel_deltas[out_ch_cnt][in_ch_cnt][ky][kx]) + (momentum * change_weights[out_ch_cnt][in_ch_cnt][ky][kx]);
                    kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx] += change_weights[out_ch_cnt][in_ch_cnt][ky][kx];
                }
            }
        }
    }
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

/*
#include <iostream>
#include <chrono>

// Function 1
void convolution::conv_forward1()
{
    // implementation of function 1
}

// Function 2
void convolution::conv_forward2()
{
    // implementation of function 2
}

int main()
{
    // number of times to run each function
    int num_runs = 10;

    // time variables
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds1, elapsed_seconds2;

    // run function 1 and record runtime
    start = std::chrono::system_clock::now();
    for (int i = 0; i < num_runs; i++)
    {
        convolution::conv_forward1();
    }
    end = std::chrono::system_clock::now();
    elapsed_seconds1 = end - start;

    // run function 2 and record runtime
    start = std::chrono::system_clock::now();
    for (int i = 0; i < num_runs; i++)
    {
        convolution::conv_forward2();
    }
    end = std::chrono::system_clock::now();
    elapsed_seconds2 = end - start;

    // print results
    std::cout << "Function 1 took " << elapsed_seconds1.count() << " seconds for " << num_runs << " runs." << std::endl;
    std::cout << "Function 2 took " << elapsed_seconds2.count() << " seconds for " << num_runs << " runs." << std::endl;

    // choose faster function and use it for further processing
    if (elapsed_seconds1 < elapsed_seconds2)
    {
        std::cout << "Function 1 is faster." << std::endl;
        convolution::conv_forward1();
    }
    else
    {
        std::cout << "Function 2 is faster." << std::endl;
        convolution::conv_forward2();
    }

    return 0;
}

*/