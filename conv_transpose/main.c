#include <iostream>
#include <vector>

using namespace std;

// Transpose convolution operation
vector<vector<vector<double>>> transposeConvolution(
    vector<vector<vector<double>>> input,
    vector<vector<vector<vector<double>>>> kernel,
    int stride)
{
    int input_channels = input.size();
    int output_channels = kernel[0].size();
    int kernel_size = kernel[0][0].size();
    int output_size = (input[0][0].size() - 1) * stride + kernel_size;


    // Initialize output tensor with zeros
    vector<vector<vector<double>>> output(output_channels, 
                                           vector<vector<double>>(output_size, 
                                                                   vector<double>(output_size)));

    // Perform transpose convolution operation
    for (int oc = 0; oc < output_channels; oc++) {
        for (int ic = 0; ic < input_channels; ic++) {
            for (int yo = 0; yo < output_size; yo += stride) {
                for (int xo = 0; xo < output_size; xo += stride) {
                    for (int yk = 0; yk < kernel_size; yk++) {
                        for (int xk = 0; xk < kernel_size; xk++) {
                            int yi = yo + yk - kernel_size + 1;
                            int xi = xo + xk - kernel_size + 1;
                            cout << " oc = " << oc << " yo = " << yo << " xo = " << xo << " ic = " << ic << " yi = " << yi << " xi = " << xi << " ic = " << ic << " oc = " << oc << " yk = " << yk << " xk = " << xk << endl;

                            if (xi >= 0 && xi < input[ic][0].size() && yi >= 0 && yi < input[ic].size()) {
                                output[oc][yo][xo] += input[ic][yi][xi] * kernel[ic][oc][yk][xk];
                            cout << "Operation at oc = " << oc << " yo = " << yo << " xo = " << xo << " ic = " << ic << " yi = " << yi << " xi = " << xi << " ic = " << ic << " oc = " << oc << " yk = " << yk << " xk = " << xk << endl;

                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

int main()
{
    // Set up input tensor
    int input_channels = 3;
    int input_size = 10;
    vector<vector<vector<double>>> input(input_channels,
                                          vector<vector<double>>(input_size, 
                                                                  vector<double>(input_size)));
    for (int ic = 0; ic < input_channels; ic++) {
        for (int yi = 0; yi < input_size; yi++) {
            for (int xi = 0; xi < input_size; xi++) {
                input[ic][yi][xi] = xi * yi;
            }
        }
    }

    // Set up kernel tensor
    int output_channels = 8;
    int kernel_size = 5;
    vector<vector<vector<vector<double>>>> kernel(input_channels,
                                                    vector<vector<vector<double>>>(output_channels,
                                                                                    vector<vector<double>>(kernel_size,
                                                                                                            vector<double>(kernel_size))));
    for (int ic = 0; ic < input_channels; ic++) {
        for (int oc = 0; oc < output_channels; oc++) {
            for (int yk = 0; yk < kernel_size; yk++) {
                for (int xk = 0; xk < kernel_size; xk++) {
                    kernel[ic][oc][yk][xk] = xk * yk;
                }
            }
        }
    }

    // Perform transpose convolution operation with stride = 2
    int stride = 2;
    vector<vector<vector<double>>> output = transposeConvolution(input, kernel, stride);
    int output_size = (input[0][0].size() - 1) * stride + kernel_size;
    // Print output
    for (int i = 0; i < output_channels; i++)
    {
        cout << "Output Channel " << i + 1 << ":" << endl;
        for (int j = 0; j < output_size; j++)
        {
            for (int k = 0; k < output_size; k++)
            {
                cout << output[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}
