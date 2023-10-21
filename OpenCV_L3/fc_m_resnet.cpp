#include "fc_m_resnet.hpp"
#include <iostream>
#include <fstream>
#include <stdlib.h> // for exit(0)
#include <math.h>
#include <time.h>

using namespace std;

fc_m_resnet::fc_m_resnet(/* args */)
{
    version_major = 0;
    version_mid = 1;
    version_minor = 1;
    // 0.0.4 fix softmax bugs
    // 0.0.5 fix bug when block type < 2 remove loss calclulation in backprop if not end block
    // 0.0.6 fix bug at  if (block_type < 2){} add else{ .... for end block } at  void fc_m_resnet::set_nr_of_hidden_nodes_on_layer_nr(int nodes)
    // before 0.0.6 (block_type < 2){ use_skip_connect_mode = 0; } bug always remove use_skip_connect_mode = 1
    // 0.0.7 use_skip_connect_mode = 0 forced on top block as well as end block. Remove several printout ==== Skip connection is used ====
    // 0.0.8 fix use_skip_connect_mode printout no mater if (inp_l_size != out_l_size)
    // 0.1.0 "shift_ununiform_skip_connection_after_samp_n" are introduced when ununifor skip connections is the case.
    // 0.1.1 loss_A and loss_B
    shift_ununiform_skip_connection_after_samp_n = 0;
    shift_ununiform_skip_connection_sample_counter = 0;
    switch_skip_con_selector = 0;

    setup_state = 0;
    nr_of_hidden_layers = 0;
    setup_inc_layer_cnt = 0;
    // 0 = start up state, nothing done yet
    // 1 = set_nr_of_hidden_layers() is set up
    // 2 = set_nr_of_hid_nodeson_layer() is done
    // 3 = init_weights or load weights is done
    block_type = 0; // 0..2
    // 0 = start block means no i_layer_delta i produced
    // 1 = middle block means both i_layer_delta is produced (backpropagate) and o_layer_delta is needed
    // 2 = end block. target_nodes is used but o_layer_delta not used only loss and i_layer_delta i calculated.
    use_softmax = 0; //
    // 0 = No softmax
    // 1 = Use softmax at end. Only possible to enable if block_type = 2 end block
    activation_function_mode = 0;
    // 0 = sigmoid activation function
    // 1 = Relu simple activation function
    // 2 = Relu fix leaky activation function
    // 3 = Relu random variable leaky activation function
    fix_leaky_proportion = 0.05;
    use_skip_connect_mode = 0;
    // 0 = turn OFF skip connections, ordinary fully connected nn block only
    // 1 = turn ON skip connectons
    skip_conn_rest_part = 0;
    skip_conn_multiple_part = 0;
    skip_conn_in_out_relation = 0;
    // 0 = same input/output
    // 1 = input > output
    // 2 = output > input
    training_mode = 0;
    // 0 = SGD Stocastic Gradient Decent
    // 1 = Batch Gradient Decent, not yet implemented
    use_dropouts = 0;
    // 0 = No dropout
    // 1 = Use dropout
    batch_size = 0; // Only used if trainging_mode 1
    loss_A = 0.0;
    loss_B = 0.0;
    learning_rate = 0.0;
    momentum = 0.0;
    dropout_proportion = 0.0;
    use_dropouts = 0;
    cout << "fc_m_resnet Constructor" << endl;
    srand(time(NULL)); // Seed radomizer
    cout << "Seed radomizer done" << endl;
}

fc_m_resnet::~fc_m_resnet()
{
    cout << "fc_m_resnet Destructor" << endl;
}
void fc_m_resnet::get_version(void)
{
    cout << "Fully connected residual neural network object" << endl;
    cout << "fc_m_resnet object version : " << version_major << "." << version_mid << "." << version_minor << endl;
    ver_major = version_major;
    ver_mid = version_mid;
    ver_minor = version_minor;
}
void fc_m_resnet::set_nr_of_hidden_layers(int nr_of_hid_layers)
{
    cout << endl;
    cout << endl;

    const int MAX_LAYERS = 100;
    if (nr_of_hid_layers < 1)
    {
        cout << "ERROR! Setup error, nr_of_hid_layers < 1. At least 1 hidden layer must be used in this architecture. nr_of_hid_layers = " << nr_of_hid_layers << endl;
        cout << "Set nr_of_hid_layers in range 1 to MAX_LAYERS = " << MAX_LAYERS << endl;
        cout << "Exit program !" << endl;
        exit(0);
    }
    else
    {
        if (nr_of_hid_layers > MAX_LAYERS)
        {
            cout << "ERROR! Setup error, nr_of_hid_layers is set too LARGE ! nr_of_hid_layers = " << nr_of_hid_layers << endl;
            cout << "MAX_LAYERS = " << MAX_LAYERS << endl;
            cout << "Exit program !" << endl;
            exit(0);
        }
        if (setup_state != 0)
        {
            cout << "ERROR! Setup error, maybe call set_nr_of_hidden_layers(int) more the one time, setup_state = " << setup_state << endl;
            cout << "set_nr_of_hidden_layers(int) should only calls once " << endl;
            cout << "Exit program !" << endl;
            exit(0);
        }
        //====================== Set up vectors layers  ====================
        nr_of_hidden_layers = nr_of_hid_layers;
        cout << " Number of hidden layers is set to = " << nr_of_hidden_layers << endl;
        for (int i = 0; i < nr_of_hidden_layers; i++)
        {
            number_of_hidden_nodes.push_back(0); // Emty number of nodes yet. Then after this call is done. driver will fill this vector with numbers of nodes on each hidden layers
        }
        setup_state = 1;
        //====================== Vectors layer setup finnish =======================
    }
}
void fc_m_resnet::randomize_weights(double rand_proportion)
{
    if (setup_state < 2)
    {
        cout << "ERROR! Setup error, setup_state is < 2 when calling randomize_weights() function setup_state = " << setup_state << endl;
        cout << "Exit program !" << endl;
        exit(0);
    }
    cout << "Randomize weights 3D vector all weights of fc_resnet object...." << endl;
    int layers = all_weights.size();
    for (int l_cnt = 0; l_cnt < layers; l_cnt++)
    {
        int nodes_on_this_layer = all_weights[l_cnt].size();
        for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
        {
            int weights_to_this_node = all_weights[l_cnt][n_cnt].size();
            for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
            {
                all_weights[l_cnt][n_cnt][w_cnt] = ((((double)rand() / RAND_MAX) - 0.5) * rand_proportion);
            }
        }
    }
    if (setup_state == 2)
    {
        setup_state = 3;
    }
    cout << "Randomize weights is DONE!" << endl;
    cout << "setup_state = " << setup_state << endl;
}

const int precision_to_text_file = 16;
void fc_m_resnet::save_weights(string filename)
{
    if (setup_state > 2)
    {
        cout << "Save data weights ..." << endl;
        ofstream outputFile;
        outputFile.precision(precision_to_text_file);
        outputFile.open(filename);
        int layers = all_weights.size();
        for (int l_cnt = 0; l_cnt < layers; l_cnt++)
        {
            int nodes_on_this_layer = all_weights[l_cnt].size();
            for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
            {
                int weights_to_this_node = all_weights[l_cnt][n_cnt].size();
                for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
                {
                    outputFile << all_weights[l_cnt][n_cnt][w_cnt] << endl;
                }
            }
        }
        outputFile.close();
        cout << "Save data finnish !" << endl;
    }
    else
    {
        cout << "ERROR! Setup error. weights file can only be saved when setup_state is in mode > 2. setup_state = " << setup_state << endl;
        cout << "Exit program !" << endl;
        exit(0);
    }
}
void fc_m_resnet::load_weights(string filename)
{
    if (setup_state < 2)
    {
        cout << "ERROR! Setup error, setup_state is < 2 when calling load_weights() function setup_state = " << setup_state << endl;
        cout << "Exit program !" << endl;
        exit(0);
    }
    else
    {
        cout << "Load data weights ..." << endl;
        ifstream inputFile;
        inputFile.precision(precision_to_text_file);
        inputFile.open(filename);
        double d_element = 0.0;
        int data_load_error = 0;
        int layers = all_weights.size();
        int data_load_numbers = 0;
        for (int l_cnt = 0; l_cnt < layers; l_cnt++)
        {
            int nodes_on_this_layer = all_weights[l_cnt].size();
            for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
            {
                int weights_to_this_node = all_weights[l_cnt][n_cnt].size();
                for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
                {
                    if (inputFile >> d_element)
                    {
                        all_weights[l_cnt][n_cnt][w_cnt] = d_element;
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
        inputFile.close();
        if (data_load_error == 0)
        {
            cout << "Load data finnish !" << endl;
        }
        else
        {
            cout << "ERROR! weight file error have not sufficient amount of data to put into  all_weights[l_cnt][n_cnt][w_cnt] vector" << endl;
            cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
        }
        setup_state = 3;
    }
}

void fc_m_resnet::set_nr_of_hidden_nodes_on_layer_nr(int nodes)
{
    if (setup_state < 1)
    {
        cout << "ERROR! Setup error. number of hidden layer is not set" << endl;
        cout << "Exit program !" << endl;
        exit(0);
    }
    if (setup_state > 1)
    {
        cout << "ERROR! Setup error. All hidden layers is alread setup, setup_state > 1" << endl;
        cout << "Exit program !" << endl;
        exit(0);
    }

    // ========= Check i_ and o_   layer_delta.size match input and output_layer size ==========
    // 0 = start block means no i_layer_delta i produced
    // 1 = middle block means both i_layer_delta is produced (backpropagate) and o_layer_delta is needed
    // 2 = end block. target_nodes is used but o_layer_delta not used only loss and i_layer_delta i calculated.
    if (block_type > 0)
    {
        // not start block then we check size of i_layer_delta
        if (i_layer_delta.size() != input_layer.size())
        {
            cout << "ERROR! Setup error. i_layer_delta.size() != input_layer.size() " << endl;
            cout << "setup all public vector size before call set_nr_of_hidden_nodes_on_layer_nr()" << endl;
            cout << "Exit program !" << endl;
            exit(0);
        }
    }
    else
    {
        if (use_skip_connect_mode == 1)
        {
            cout << "ERROR! Setup error. use_skip_connect_mode ON is not allowed at top block" << endl;
            cout << "Note use_skip_connect_mode is now FORCED to 0 instead !!" << endl;
            use_skip_connect_mode = 0;
            cout << "use_skip_connect_mode = " << use_skip_connect_mode << endl;
        }
    }
    if (block_type < 2)
    {
        // not end block the we check size of i_layer_delta
        if (o_layer_delta.size() != output_layer.size())
        {
            cout << "ERROR! Setup error. o_layer_delta.size() != output_layer.size() " << endl;
            cout << "setup all public vector size before call set_nr_of_hidden_nodes_on_layer_nr()" << endl;
            cout << "Exit program !" << endl;
            exit(0);
        }
        if (use_softmax == 1)
        {
            cout << "ERROR! Setup error. Sofmax is only allowed at End block" << endl;
            cout << "Tip ! Set use_softmax = 0 when not using End block block_type < 2" << endl;
            cout << "Exit program !" << endl;
            exit(0);
        }
    }
    else
    {
        if (use_skip_connect_mode == 1)
        {
            cout << "ERROR! Setup error. use_skip_connect_mode ON is not allowed at End block" << endl;
            cout << "Note use_skip_connect_mode is now FORCED to 0 instead !!" << endl;
            use_skip_connect_mode = 0;
            cout << "use_skip_connect_mode = " << use_skip_connect_mode << endl;
        }
    }
    //=============================================================================================
    vector<double> dummy_hidd_nodes_on_this_layer;
    for (int i = 0; i < nodes; i++)
    {
        dummy_hidd_nodes_on_this_layer.push_back(0.0);
    }
    hidden_layer.push_back(dummy_hidd_nodes_on_this_layer);
    internal_delta.push_back(dummy_hidd_nodes_on_this_layer);

    number_of_hidden_nodes[setup_inc_layer_cnt] = nodes;
    cout << "Size of hidden_layer[" << setup_inc_layer_cnt << "][x] = " << hidden_layer[setup_inc_layer_cnt].size() << endl;

    setup_inc_layer_cnt++;
    if (setup_inc_layer_cnt > nr_of_hidden_layers - 1)
    {
        cout << "hidden_layer vector is now set up" << endl;
        int out_layer_size = output_layer.size();
        vector<double> temporary_dummy_output_vector;
        for (int i = 0; i < out_layer_size; i++)
        {
            temporary_dummy_output_vector.push_back(0.0);
        }
        internal_delta.push_back(temporary_dummy_output_vector);

        // i_layer_delta and o_layer_delta data will NOT copy to last and first internal_delta data this are diffrent data
        // i_layer_delta is the delta data at the end point of thi input wires of this nn block
        // internal_delta[0][x] will contain the delta at the hildden_layer[0][x]
        // i_layer_delta[x] will contain the delta backproped from internal_delta[0][x] via first layer wheights and skip wire sum at top
        // Likevice internal_delta[end output layer][x] before the activation function
        // but o_layer_delta[x] is the delta from outside after the activation function

        cout << "Now setup all_weight, change_weights vectors size of this fc block" << endl;
        vector<double> dummy_1D_weight_vector;
        vector<vector<double>> dummy_2D_weight_vector;
        int weight_size_1D = 0;
        int weight_size_2D = 0;
        for (int l_cnt = 0; l_cnt < nr_of_hidden_layers; l_cnt++)
        {
            dummy_1D_weight_vector.clear();
            dummy_2D_weight_vector.clear();
            if (l_cnt == 0)
            {
                // Input layer connect to hidden_layer[0]
                weight_size_1D = input_layer.size() + 1; // +1 for the Bias wheight
                for (int i = 0; i < weight_size_1D; i++)
                {
                    dummy_1D_weight_vector.push_back(0.0);
                }

                cout << "Size of temporary dummy_1D_weight_vector from input layer connection[" << l_cnt << "] is = " << dummy_1D_weight_vector.size() << endl;
            }
            else
            {
                // weight size make up for connections between hidden_layer before to this hiden_layer
                for (int i = 0; i < number_of_hidden_nodes[l_cnt - 1] + 1; i++) // +1 for Bias weight
                {
                    dummy_1D_weight_vector.push_back(0.0);
                }
                cout << "Size of temporary dummy_1D_weight_vector from hidden layer connection[" << l_cnt << "] is = " << dummy_1D_weight_vector.size() << endl;
            }
            weight_size_2D = hidden_layer[l_cnt].size();
            for (int n_cnt = 0; n_cnt < weight_size_2D; n_cnt++)
            {
                dummy_2D_weight_vector.push_back(dummy_1D_weight_vector);
            }
            all_weights.push_back(dummy_2D_weight_vector);
            change_weights.push_back(dummy_2D_weight_vector);
        }

        // output weight setup
        weight_size_1D = number_of_hidden_nodes[nr_of_hidden_layers - 1] + 1; // +1 is because that we have 1 weights added for the Bias node
        dummy_1D_weight_vector.clear();
        for (int i = 0; i < weight_size_1D; i++) //
        {
            dummy_1D_weight_vector.push_back(0.0);
        }
        cout << "Size of temporary dummy_1D_weight_vector last hidden layer connection[" << nr_of_hidden_layers - 1 << "] is = " << dummy_1D_weight_vector.size() << endl;

        weight_size_2D = output_layer.size();
        dummy_2D_weight_vector.clear();
        for (int n_cnt = 0; n_cnt < weight_size_2D; n_cnt++)
        {
            dummy_2D_weight_vector.push_back(dummy_1D_weight_vector);
        }
        all_weights.push_back(dummy_2D_weight_vector);
        change_weights.push_back(dummy_2D_weight_vector);
        cout << "The size of all_weight and change_weights in now setup OK !" << endl;
        cout << "Note that the program how call this object could only set this size once. No protections against change size of the public vectors" << endl;
        setup_state = 2;
        cout << "Setup state = " << setup_state << endl;
        // Print out the size of the network structure
        int weight_l_size = all_weights.size();
        cout << "Size of layer dimentions[] of weights at the this nn block = " << weight_l_size << endl;
        for (int i = 0; i < weight_l_size; i++)
        {
            int weight_n_size = all_weights[i].size();
            int weight_w_size = all_weights[i][weight_n_size - 1].size();
            cout << "Size of node dimentions[][] of weights for hidden layer number " << i << " is: " << weight_n_size << endl;
            cout << "Size of weight dimentions[][][] of weights for hidden layer number " << i << " is: " << weight_w_size << endl;
        }
    }
    if (use_skip_connect_mode == 1)
    {
        int inp_l_size = input_layer.size();
        int out_l_size = output_layer.size();
        // skip_conn_in_out_relation
        // 0 = same input/output
        // 1 = input > output
        // 2 = output > input
        if (inp_l_size != out_l_size)
        {
            if (inp_l_size > 0 && out_l_size > 0)
            {
                skip_conn_rest_part = inp_l_size % out_l_size;
                if (inp_l_size > out_l_size)
                {
                    skip_conn_in_out_relation = 1; // 1 = input > output
                    skip_conn_multiple_part = inp_l_size / out_l_size;
                }
                else
                {
                    skip_conn_in_out_relation = 2; // 2 = output > input
                    skip_conn_multiple_part = out_l_size / inp_l_size;
                }
            }
            else
            {
                cout << "Error input_layer.size() or output_layer.size() = 0 " << endl;
                cout << "input_layer.size() = " << input_layer.size() << endl;
                cout << "output_layer.size() = " << output_layer.size() << endl;
                cout << "Exit program " << endl;
                exit(0);
            }
        }
        else
        {
            // input out are symetric
            skip_conn_in_out_relation = 0; // 0 = same input/output
            skip_conn_multiple_part = 1;
            skip_conn_rest_part = 0;
            if (shift_ununiform_skip_connection_after_samp_n < 1)
            {
                cout << "Information: skip connections is uniform on this block but shift_ununiform_skip_connection_after_samp_n < 1 " << endl;
                cout << "shift_ununiform_skip_connection_after_samp_n = " << shift_ununiform_skip_connection_after_samp_n << " will not have any affect" << endl;
            }
        }
        if (setup_inc_layer_cnt == 1) // Only print out 1 time
        {
            cout << "==== Skip connection is used ====" << endl;
            cout << "input_layer.size() = " << input_layer.size() << endl;
            cout << "output_layer.size() = " << output_layer.size() << endl;
            cout << "skip_conn_multiple_part = " << skip_conn_multiple_part << endl;
            cout << "skip_conn_rest_part = " << skip_conn_rest_part << endl;
        }
    }
}
double fc_m_resnet::activation_function(double input_data, int end_layer)
{
    double output_data = 0.0;
    int this_node_dopped_out = 0;
    if (use_dropouts == 1 && end_layer == 0)
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
double fc_m_resnet::delta_activation_func(double delta_outside_function, double value_from_node_outputs)
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
void fc_m_resnet::forward_pass(void)
{
    for (int l_cnt = 0; l_cnt < nr_of_hidden_layers; l_cnt++) // Loop though all hidden layers
    {
        int dst_nodes = hidden_layer[l_cnt].size(); //
        if (l_cnt == 0)
        {
            // Input layer will connect to first hidden layer
            int src_nodes = input_layer.size();
            for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
            {
                double acc_dot_product = all_weights[l_cnt][dst_n_cnt][src_nodes]; // Set the bias weight as the start value
                for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
                {
                    acc_dot_product += input_layer[src_n_cnt] * all_weights[l_cnt][dst_n_cnt][src_n_cnt]; // Add all to make the dot product
                }
                // Dot product finnished
                // Make the activation function
                hidden_layer[l_cnt][dst_n_cnt] = activation_function(acc_dot_product, 0);
            }
        }
        else
        {
            // Connection from hidden layer to next hidden layer
            int src_nodes = hidden_layer[l_cnt - 1].size(); // Note -1 because we take the signal from the hidden layer befor this desitnatiuon layer we will calculate at this l_cnt.
            for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
            {
                double acc_dot_product = all_weights[l_cnt][dst_n_cnt][src_nodes]; // Set the bias weight as the start value
                for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
                {
                    acc_dot_product += hidden_layer[l_cnt - 1][src_n_cnt] * all_weights[l_cnt][dst_n_cnt][src_n_cnt]; // Add all to make the dot product
                }
                // Dot product finnished
                // Make the activation function
                hidden_layer[l_cnt][dst_n_cnt] = activation_function(acc_dot_product, 0);
            }
        }
    }

    // Last hidden layer will be connected to output layer
    int dst_nodes = output_layer.size();
    int l_cnt = hidden_layer.size();
    int last_hidden_layer_nr = l_cnt - 1;
    int src_nodes = hidden_layer[last_hidden_layer_nr].size();
    for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
    {
        double acc_dot_product = all_weights[l_cnt][dst_n_cnt][src_nodes]; // Set the bias weight as the start value
        for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
        {
            acc_dot_product += hidden_layer[l_cnt - 1][src_n_cnt] * all_weights[l_cnt][dst_n_cnt][src_n_cnt]; // Add all to make the dot product
        }
        // Dot product finnished

        if (use_softmax == 1)
        {
            output_layer[dst_n_cnt] = acc_dot_product; // Do softmax outside this for loop
        }
        else
        {
            // Make the activation function
            output_layer[dst_n_cnt] = activation_function(acc_dot_product, 1);
        }
    }

    if (use_softmax == 1)
    {
        // Make softmax activation function
        double sum_exp_input = 0.0;
        for (int out_cnt = 0; out_cnt < dst_nodes; out_cnt++)
        {
            output_layer[out_cnt] = exp(output_layer[out_cnt]);
            sum_exp_input += output_layer[out_cnt];
            if (sum_exp_input == 0.0)
            {
                // Zero div protection
                sum_exp_input = 0.000000000000001;
            }
        }
        for (int out_cnt = 0; out_cnt < dst_nodes; out_cnt++)
        {
            output_layer[out_cnt] = output_layer[out_cnt] / sum_exp_input;
        }
    }

    if (use_skip_connect_mode == 1 && use_softmax == 0)
    {
        int src_nodes = input_layer.size();
        int dst_nodes = output_layer.size();

        if (skip_conn_in_out_relation == 0)
        {
            // 0 = same input/output
            for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
            {
                output_layer[dst_n_cnt] += input_layer[dst_n_cnt]; // Input nodes are same as output nodes simple add operation at output side
            }
        }
        else
        {

            if (shift_ununiform_skip_connection_after_samp_n == 0)
            {
                // Allways connect all ununiform skip connections mode
                if (skip_conn_in_out_relation == 1)
                {
                    // 1 = input > output

                    for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
                    {
                        output_layer[src_n_cnt % dst_nodes] += input_layer[src_n_cnt]; // Input nodes are > output nodes
                    }
                }
                else if (skip_conn_in_out_relation == 2)
                {
                    // 2 = output > input
                    for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
                    {
                        output_layer[dst_nodes] += input_layer[dst_n_cnt % src_nodes]; // Input nodes are < output nodes
                                                                                       //          cout << "skip_conn_in_out_relation = " << skip_conn_in_out_relation << " dst_nodes = " << dst_nodes << " dst_n_cnt % src_nodes" << dst_n_cnt % src_nodes <<endl;
                    }
                }
            }
            else
            {
                // Use the switch skip connection mode

                if (shift_ununiform_skip_connection_sample_counter < shift_ununiform_skip_connection_after_samp_n)
                {
                    shift_ununiform_skip_connection_sample_counter++;
                }
                else
                {
                    if (switch_skip_con_selector < skip_conn_multiple_part - 1)
                    {
                        switch_skip_con_selector++; //
                    }
                    else
                    {
                        switch_skip_con_selector = 0;
                    }
                    shift_ununiform_skip_connection_sample_counter = 0;
                }
                if (skip_conn_in_out_relation == 1)
                {
                    // 1 = input > output
                    for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
                    {
                        output_layer[dst_n_cnt] += input_layer[dst_n_cnt + dst_nodes * switch_skip_con_selector]; // Input nodes are > output nodes
                    }
                }
                else if (skip_conn_in_out_relation == 2)
                {
                    // 2 = output > input
                    for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
                    {
                        output_layer[src_n_cnt + src_nodes * switch_skip_con_selector] += input_layer[src_n_cnt]; // Input nodes are < output nodes
                    }
                }
            }
        }
    }
}

void fc_m_resnet::only_loss_calculation(void)
{
    int nr_out_nodes = output_layer.size();
    for (int i = 0; i < nr_out_nodes; i++)
    {
        loss_A += 0.5 * (target_layer[i] - output_layer[i]) * (target_layer[i] - output_layer[i]); // Squared error * 0.5
        loss_B += 0.5 * (target_layer[i] - output_layer[i]) * (target_layer[i] - output_layer[i]); // Squared error * 0.5
    }
}
void fc_m_resnet::backpropagtion_and_update(void)
{

    int output_nodes = output_layer.size();
    if (block_type == 2)
    {
        for (int dst_cnt = 0; dst_cnt < output_nodes; dst_cnt++)
        {
            if (use_softmax == 0)
            {
                double loss_calc = 0.5 * (target_layer[dst_cnt] - output_layer[dst_cnt]) * (target_layer[dst_cnt] - output_layer[dst_cnt]); // Squared error * 0.5
                loss_A += loss_calc; //
                loss_B += loss_calc;//
            }
            else
            {
                double loss_calc = target_layer[dst_cnt] * log(output_layer[dst_cnt]);
                loss_A -=  loss_calc;
                loss_B -=  loss_calc;
            }
        }
    }

    //============ Calculated and Backpropagate output delta and neural network loss ============
    int nr_out_nodes = output_layer.size();
    int last_delta_layer_nr = internal_delta.size() - 1;

    for (int i = 0; i < nr_out_nodes; i++)
    {
        if (block_type == 2)
        {
            if (use_softmax == 0)
            {
                internal_delta[last_delta_layer_nr][i] = delta_activation_func((target_layer[i] - output_layer[i]), output_layer[i]);
            }
            else
            {
                internal_delta[last_delta_layer_nr][i] = (target_layer[i] - output_layer[i]);
            }
        }
        else
        {
            internal_delta[last_delta_layer_nr][i] = delta_activation_func(o_layer_delta[i], output_layer[i]);
        }
    }
    //============================================================================================

    //============ Backpropagate hidden layer errors ============
    for (int i = last_delta_layer_nr - 1; i > -1; i--) // last_delta_layer_nr-1 (-1) because last layer delta already calculated for output layer laready cacluladed above
    {
        int nr_delta_nodes_dst_layer = internal_delta[i].size();
        int nr_delta_nodes_src_layer = internal_delta[i + 1].size();
        for (int dst_n_cnt = 0; dst_n_cnt < nr_delta_nodes_dst_layer; dst_n_cnt++)
        {
            double accumulated_backprop = 0.0;
            for (int src_n_cnt = 0; src_n_cnt < nr_delta_nodes_src_layer; src_n_cnt++)
            {
                accumulated_backprop += all_weights[i + 1][src_n_cnt][dst_n_cnt] * internal_delta[i + 1][src_n_cnt];
            }
            internal_delta[i][dst_n_cnt] = delta_activation_func(accumulated_backprop, hidden_layer[i][dst_n_cnt]);
        }
    }

    //============ Backpropagate i_layer_delta ============
    if (block_type > 0) // Skip backprop to i_layer_delta if this object is a start block. 0 = start block
    {
        int nr_delta_nodes_dst_layer = i_layer_delta.size();
        int nr_delta_nodes_src_layer = internal_delta[0].size();
        for (int dst_n_cnt = 0; dst_n_cnt < nr_delta_nodes_dst_layer; dst_n_cnt++)
        {
            double accumulated_backprop = 0.0;
            for (int src_n_cnt = 0; src_n_cnt < nr_delta_nodes_src_layer; src_n_cnt++)
            {
                accumulated_backprop += all_weights[0][src_n_cnt][dst_n_cnt] * internal_delta[0][src_n_cnt];
            }
            i_layer_delta[dst_n_cnt] = accumulated_backprop;
        }
        if (use_skip_connect_mode == 1 && use_softmax == 0)
        {
            // cout << "debug1" << endl;
            int src_nodes = input_layer.size();
            int dst_nodes = output_layer.size();
            if (skip_conn_in_out_relation == 0)
            {
                // 0 = same input/output
                for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
                {
                    i_layer_delta[dst_n_cnt] += o_layer_delta[dst_n_cnt]; // Input nodes are same as output nodes simple add operation at output side
                }
            }
            else if (skip_conn_in_out_relation == 1)
            {

                // 1 = input > output
                if (shift_ununiform_skip_connection_after_samp_n == 0)
                {
                    for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
                    {
                        i_layer_delta[src_n_cnt] += o_layer_delta[src_n_cnt % dst_nodes]; // Input nodes are > output nodes
                    }
                }
                else
                {
                    for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
                    {
                        i_layer_delta[dst_n_cnt + dst_nodes * switch_skip_con_selector] += o_layer_delta[dst_n_cnt]; // Input nodes are > output nodes
                    }
                }
            }
            else if (skip_conn_in_out_relation == 2)
            {
                // 2 = output > input
                if (shift_ununiform_skip_connection_after_samp_n == 0)
                {

                    for (int dst_n_cnt = 0; dst_n_cnt < dst_nodes; dst_n_cnt++)
                    {
                        i_layer_delta[dst_n_cnt % src_nodes] += o_layer_delta[dst_nodes]; //
                    }
                }
                else
                {
                    for (int src_n_cnt = 0; src_n_cnt < src_nodes; src_n_cnt++)
                    {
                        i_layer_delta[src_n_cnt] += o_layer_delta[src_n_cnt + src_nodes * switch_skip_con_selector]; // Input nodes are > output nodes
                    }
                }
            }
        }
    }

    //============ Backpropagate finish =================================

    // ======== Update weights ========================
    int weight_size_1D = all_weights.size();
    for (int i = 0; i < weight_size_1D; i++)
    {

        int weight_size_2D = all_weights[i].size();
        if (i == 0)
        {
            // Input layer
            for (int j = 0; j < weight_size_2D; j++)
            {
                int weight_size_3D = all_weights[i][j].size() - 1;
                change_weights[i][j][weight_size_3D] = learning_rate * internal_delta[i][j] + momentum * change_weights[i][j][weight_size_3D]; // Bias weight
                all_weights[i][j][weight_size_3D] += change_weights[i][j][weight_size_3D];                                                     // Update Bias wheight
                for (int k = 0; k < weight_size_3D; k++)
                {
                    change_weights[i][j][k] = learning_rate * input_layer[k] * internal_delta[i][j] + momentum * change_weights[i][j][k];
                    all_weights[i][j][k] += change_weights[i][j][k];
                }
            }
        }
        else
        {
            // Hidden layer
            for (int j = 0; j < weight_size_2D; j++)
            {
                int weight_size_3D = all_weights[i][j].size() - 1;
                change_weights[i][j][weight_size_3D] = learning_rate * internal_delta[i][j] + momentum * change_weights[i][j][weight_size_3D]; // Bias weight
                all_weights[i][j][weight_size_3D] += change_weights[i][j][weight_size_3D];                                                     // Update Bias wheight
                for (int k = 0; k < weight_size_3D; k++)
                {
                    change_weights[i][j][k] = learning_rate * hidden_layer[i - 1][k] * internal_delta[i][j] + momentum * change_weights[i][j][k];
                    all_weights[i][j][k] += change_weights[i][j][k];
                }
            }
        }
    }
    // ===============================================
}
void fc_m_resnet::print_weights(void)
{
    cout << "all_weights = " << endl;
    int weight_size_1D = all_weights.size();
    for (int i = 0; i < weight_size_1D; i++)
    {
        int weight_size_2D = all_weights[i].size();
        for (int j = 0; j < weight_size_2D; j++)
        {
            int weight_size_3D = all_weights[i][j].size();
            for (int k = 0; k < weight_size_3D; k++)
            {
                cout << " " << all_weights[i][j][k];
            }
            cout << endl;
        }
        cout << endl;
    }
}

//==== Used or verify gradient calcualtion ================================================================
//================= Functions only for debugging/verify the backpropagation gradient functions ============
// You must turn OFF drop out and set learning rate to 0.0 when verify backpropagation
double fc_m_resnet::verify_gradient(int l, int n, int w, double adjust_weight)
{
    double gradient_return = 0.0;
    all_weights[l][n][w] += adjust_weight;
    if (l > 0)
    {
        // from hidden layer
        gradient_return = hidden_layer[l - 1][w] * internal_delta[l][n];
    }
    else
    {
        // from input layer
        gradient_return = input_layer[w] * internal_delta[l][n];
    }
    return gradient_return;
}
//================= Functions only for debugging/verify the backpropagation gradient functions ============
// You must turn OFF drop out and set learning rate to 0.0 when verify backpropagation
double fc_m_resnet::calc_error_verify_grad(void)
{
    int nr_out_nodes = output_layer.size();
    double error = 0;
    for (int i = 0; i < nr_out_nodes; i++)
    {
        if (block_type == 2)
        {
            error += target_layer[i] - output_layer[i];
        }
        else
        {
            error += o_layer_delta[i] - output_layer[i];
        }
        // Softmax not yet implemented
    }
    return error;
}
//=========================================================================================================
//=========================================================================================================