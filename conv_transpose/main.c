#include <iostream>
#include <vector>

using namespace std;

// Function to perform transpose convolution operation
vector<vector<vector<double>>> transposeConvolution(vector<vector<vector<double>>> input, 
                                                    vector<vector<vector<double>>> kernel,
                                                    int stride) {
    int input_channels = input.size();
    int output_channels = kernel[0].size();
    int kernel_size = kernel[0][0].size();
    int input_size = input[0].size();
    int output_size = (input_size - 1) * stride + kernel_size;

    // Initialize output vector
    vector<vector<vector<double>>> output(output_channels, 
                                           vector<vector<double>>(output_size, 
                                                                   vector<double>(output_size)));

    // Iterate over output channels
    for (int oc = 0; oc < output_channels; oc++) {
        // Iterate over input channels
        for (int ic = 0; ic < input_channels; ic++) {
            // Iterate over output spatial position
            for (int xo = 0; xo < output_size; xo++) {
                for (int yo = 0; yo < output_size; yo++) {
                    // Iterate over kernel spatial position
                    for (int xk = 0; xk < kernel_size; xk++) {
                        for (int yk = 0; yk < kernel_size; yk++) {
                            int xi = xo + xk - stride;
                            int yi = yo + yk - stride;
                            if (xi >= 0 && xi < input_size && yi >= 0 && yi < input_size) {
                                output[oc][xo][yo] += input[ic][xi][yi] * kernel[ic][oc][xk][yk];
                            }
                        }
                    }
                }
            }
        }
    }
    
    return output;
}

int main() {
    // Set arbitrary input, kernel and output sizes
    int input_channels = 3;
    int output_channels = 8;
    int kernel_size = 5;
    int input_size = 7;

    // Initialize input vector
    vector<vector<vector<double>>> input(input_channels, 
                                          vector<vector<double>>(input_size, 
                                                                  vector<double>(input_size)));
    // Initialize kernel vector
    vector<vector<vector<vector<double>>>> kernel(input_channels, 
                                                   vector<vector<vector<double>>>(output_channels, 
                                                                                   vector<vector<double>>(kernel_size, 
                                                                                                           vector<double>(kernel_size))));
    // Initialize input and kernel with arbitrary values
    for (int i = 0; i < input_channels; i++) {
        for (int j = 0; j < input_size; j++) {
            for (int k = 0; k < input_size; k++) {
                input[i][j][k] = i * j * k;
            }
        }
        for (int o = 0; o < output_channels; o++) {
            for (int x = 0; x < kernel_size; x++) {
                for (int y = 0; y < kernel_size; y++) {
                    kernel[i][o][x][y] = (i + 1) * (o + 1) * (x + 1) * (y + 1);
                }
            }
        }
    }

    // Perform transpose convolution operation with stride = 2
    int stride = 2;
    vector<vector<vector<double>>> output = transposeConvolution(input, kernel, stride);

   // Print output
    for (int i = 0; i < output_channels; i++) {
        cout << "Output Channel " << i + 1 << ":" << endl;
        for (int j = 0; j < output_size; j++) {
            for (int k = 0; k < output_size; k++) {
                cout << output[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}