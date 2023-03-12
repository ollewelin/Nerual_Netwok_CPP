/*
Description:
Regularization is a technique used to prevent overfitting, 
which occurs when a neural network memorizes the training data instead of learning to generalize.

The two most common types of regularization are L1 and L2 regularization, 
which add a penalty term to the loss function to encourage the weights to stay small. 

L1 regularization encourages sparse weights, while L2 regularization encourages small weights.
Normalization is a technique used to make the input data have zero mean and unit variance. 
Normalization is particularly useful when the input features have different scales. 
There are several types of normalization techniques, including Batch Normalization, Layer Normalization, and Instance Normalization.

Code:
This code implements the forward and backward passes of Batch Normalization. 
The batch_norm_forward function takes as input the input data x, the scaling parameters gamma and beta, 
the mean and variance of the input data, mean and var, and the number of data points N and the dimension of the input D. 
It computes the normalized input x_norm and the output out. 
The batch_norm_backward function takes as input: 
1. the gradient of the output dout, 
2. the normalized input x_norm, 
3. the scaling parameters gamma, 
4. the variance var, 
5. the number of data points N and the dimension of the input D. 

It computes the gradients of the input dx and the scaling parameters dgamma and dbeta.

You can call these functions after each convolutional layer to normalize the input data and stabilize the network during training.
*/

void batch_norm_forward(float* x, float* gamma, float* beta, float* mean, float* var, int N, int D, float* x_norm, float* out)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            x_norm[i*D + j] = (x[i*D + j] - mean[j]) / sqrt(var[j] + 1e-8);
            out[i*D + j] = gamma[j] * x_norm[i*D + j] + beta[j];
        }
    }
}

void batch_norm_backward(float* dout, float* x_norm, float* gamma, float* var, int N, int D, float* dx, float* dgamma, float* dbeta)
{
    float* dgamma_sum = (float*) calloc(D, sizeof(float));
    float* dbeta_sum = (float*) calloc(D, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            dgamma_sum[j] += dout[i*D + j] * x_norm[i*D + j];
            dbeta_sum[j] += dout[i*D + j];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            dx[i*D + j] = gamma[j] * (dout[i*D + j] - (dgamma_sum[j]*x_norm[i*D + j] + dbeta_sum[j])/(N*D)) / sqrt(var[j] + 1e-8);
        }
    }
    memcpy(dgamma, dgamma_sum, D*sizeof(float));
    memcpy(dbeta, dbeta_sum, D*sizeof(float));
    free(dgamma_sum);
    free(dbeta_sum);
}
