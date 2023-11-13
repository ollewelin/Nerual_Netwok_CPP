#include "convolution.hpp"
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

convolution::convolution()
{
    version_major = 0;
    version_mid = 3;
    version_minor = 6;
    // 0.0.0 Not finnish at all
    // 0.2.0 Added void convolution::conv_transpose_fwd() function not yet tested
    // 0.0.3 remove conv_forward2(void) function 
    // 0.3.3 Fix bug, add void convolution::clear_i_tens_delta() i_tensor_delta have not cleared data from pre sample before version 0.3.3
    // 0.3.4 Fix bug in conv_transpose_fwd() set the activation output to 1.0 instead of last forwar value
    // 0.3.5 Make kernel_weights public so it's possible for open CV to show kernels in main
    // 0.3.6 Fix bug in conv_transpose_fwd() when using Sigmoid function now add a pseudo_activation_output_value = 0.5 when sigmoid used. 1.0 when Relu used


    setup_state = 0;
    kernel_size = 3;
    stride = 1;
    dropout_proportion = 0.0;
    activation_function_mode = 0;
    use_dropouts = 0;
    top_conv = 0;
    cout << "Constructor Convloution neural network object " << endl;
    srand(time(NULL)); // Seed radomizer
    cout << "Seed radomizer done" << endl;
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
            cout << "   Note! Add bottom rows and right column at input_tensor to make the convolution between input and kernel not miss any input data during stride stop operation " << endl;
        }
        else
        {
            cout << "   OK Note. Input tensor fit perfect to stride no missing data when slide to end without add extra right rows and bottom column at input spartial " << endl;
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

    // Randomize kernel weights
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        kernel_bias_weights[out_ch_cnt] = ((((double)rand() / RAND_MAX) - 0.5) * rand_prop);
        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
        {
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx] = ((((double)rand() / RAND_MAX) - 0.5) * rand_prop);
                }
            }
        }
    }
    cout << "Randomize kernel weight data finnish !" << endl;
    setup_state = 5;
}

const int precision_to_text_file = 16;
void convolution::load_weights(string filename_k_w, string filname_k_bias)
{
    if (setup_state < 3)
    {
        cout << "Error could not load_weights() setup_state must be = 4 or more, setup_state = " << setup_state << endl;
        cout << "setup_state = " << setup_state << endl;
        cout << "Exit program" << endl;
        // TODO
        exit(0);
    }

    cout << "Load data weights ..." << endl;
    ifstream inputFile_w, inputFile_b;
    inputFile_w.precision(precision_to_text_file);
    inputFile_w.open(filename_k_w);
    inputFile_b.precision(precision_to_text_file);
    inputFile_b.open(filename_k_w);

    double d_element = 0.0;
    double d_b_element = 0.0;
    int data_load_error = 0;
    int data_load_numbers = 0;
    int data_b_load_numbers = 0;
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        if (inputFile_b >> d_b_element)
        {
            kernel_bias_weights[out_ch_cnt] = d_b_element;
            data_b_load_numbers++;
        }
        else
        {
            data_load_error = 1;
        }
        if (data_load_error != 0)
        {
            break;
        }

        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
        {
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    if (inputFile_w >> d_element)
                    {
                        kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx] = d_element;
                        data_load_numbers++;
                    }
                    else
                    {
                        data_load_error = 1;
                    }
                    if (data_load_error != 0)
                    {
                        break;
                    }
                }
                if (data_load_error != 0)
                {
                    break;
                }
            }
            if (data_load_error != 0)
            {
                break;
            }
        }
    }
    inputFile_w.close();
    inputFile_b.close();

    if (data_load_error == 0)
    {
        cout << "Load data finnish !" << endl;
    }
    else
    {
        cout << "ERROR! weight file error have not sufficient amount of data to put into  all_weights[l_cnt][n_cnt][w_cnt] vector" << endl;
        cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
        cout << "Loaded this amout of data bias weights data_load_numbers = " << data_b_load_numbers << endl;
    }
    setup_state = 5;
}
void convolution::save_weights(string filename_k_w, string filname_k_bias)
{
    cout << "Save kernel weight data weights ..." << endl;
    ofstream outputFile_w, outputFile_b;
    outputFile_w.precision(precision_to_text_file);
    outputFile_w.open(filename_k_w);
    outputFile_b.precision(precision_to_text_file);
    outputFile_b.open(filname_k_bias);
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        outputFile_b << kernel_bias_weights[out_ch_cnt] << endl;
        for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
        {
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    outputFile_w << kernel_weights[out_ch_cnt][in_ch_cnt][ky][kx] << endl;
                }
            }
        }
    }
    outputFile_b.close();
    outputFile_w.close();
    cout << "Save kernel weight data finnish !" << endl;
}

double convolution::activation_function(double input_data)
{
    double output_data = 0.0;
    int this_node_dopped_out = 0;
    if (use_dropouts == 1)
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
                            //      cout << "dot product = " << dot_product << endl;
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

void convolution::xy_start_stop_kernel(int slide_val)
{
    start_ret = slide_val % stride;
    if (slide_val > (input_side_size - kernel_size))
    {
        // Contraint start kernel pos to more then 0
        start_ret = (slide_val - (output_side_size - 1) * stride);
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

void convolution::clear_i_tens_delta()
{
    //Clear i_tensor_delta
    for (int y_slide = 0; y_slide < input_side_size; y_slide++)
    {
        for (int x_slide = 0; x_slide < input_side_size; x_slide++)
        {
            for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
            {
                // Clear data
                i_tensor_delta[in_ch_cnt][y_slide][x_slide] = 0.0;
            }
        }
    }
}

void convolution::conv_backprop()
{
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                // Clear delta for kernel weight
                for (int in_ch_cnt = 0; in_ch_cnt < input_tensor_channels; in_ch_cnt++)
                {
                    kernel_deltas[out_ch_cnt][in_ch_cnt][ky][kx] = 0.0;
                }
            }
        }
    }

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
if(top_conv != 1)
{
    //Clear i_tensor_delta first
    clear_i_tens_delta();

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
                for (int ky = y_start_ret; ky < y_stop_ret; ky += stride) // Flipped 180 deg kernel_weight
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
                    for (int kx = x_start_ret; kx < x_stop_ret; kx += stride) // Flipped 180 deg kernel_weight
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
//Exact same algorithm as convolution::conv_backprop() function but the kernel delta calculation is removed
//Notice that this function borrow same memory for traspose data as the detla memory in convolution::conv_backprop() function there for 
//You should know that delta memroy data will be overwrited here so you could not call this function before update weight if you run forward and then convolution::conv_backprop()
//You then must call this function after done forward convolution::conv_backprop() and update the weitht from conv_backprop() calculation.
//Note thet this funtion will also run if this object is a top_conv so if(top_conv != 1) condition is removed

//Use (o_tensor_delta[out_ch_cnt][y_slide][x_slide] as the input data 
//and use the i_tensor_delta[in_ch_cnt][y_slide][x_slide] as the output transpose (upsampling data output you can use for autoencoder or
//visualize at top_conv layer if you want

    //Clear i_tensor_delta first
    clear_i_tens_delta();

    // Compute delta for each output channel
    for (int out_ch_cnt = 0; out_ch_cnt < output_tensor_channels; out_ch_cnt++)
    {
        accum_bias_deltas[out_ch_cnt] = 0.0; // used only for bias weight update later
        for (int y_slide = 0; y_slide < output_side_size; y_slide++)
        {
            for (int x_slide = 0; x_slide < output_side_size; x_slide++)
            {
                // Compute derivative of activation function
                //double delta_activation = delta_activation_func(o_tensor_delta[out_ch_cnt][y_slide][x_slide], output_tensor[out_ch_cnt][y_slide][x_slide]);
                double pseudo_activation_output_value = 1.0;// 1.0 For Relu function
                if(activation_function_mode == 0)
                {
                    pseudo_activation_output_value = 0.5;// 0.5 For sigmoid function
                }
                double delta_activation = delta_activation_func(o_tensor_delta[out_ch_cnt][y_slide][x_slide], pseudo_activation_output_value);
                // delta_activation used in this loop for kernel weight delta calculations
                // store also delta_activation into internal_tensor_delta[][][] used later for delta for input tensor calculation
                internal_tensor_delta[out_ch_cnt][y_slide][x_slide] = delta_activation;
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
                for (int ky = y_start_ret; ky < y_stop_ret; ky += stride) // Flipped 180 deg kernel_weight
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
                    for (int kx = x_start_ret; kx < x_stop_ret; kx += stride) // Flipped 180 deg kernel_weight
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
