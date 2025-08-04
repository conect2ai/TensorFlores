namespace Conect2AI {
namespace TensorFlores {
class MultilayerPerceptron {
public: 

float* predict(float *x) { 
    float* y_pred = new float[1];
static const uint8_t w1[3][16] = {
    {214, 216, 209, 217, 221, 191, 205, 203, 235, 217, 222, 212, 185, 225, 215, 234},
    {225, 199, 199, 220, 207, 185, 202, 201, 183, 169, 201, 207, 210, 176, 211, 200},
    {219, 222, 230, 219, 160, 173, 139, 232, 219, 200, 230, 212, 233, 205, 218, 210}
};

static const uint8_t b1[16] = {198, 193, 189, 203, 208, 201, 209, 177, 207, 202, 203, 199, 193, 195, 196, 190};

static const uint8_t w2[16][8] = {
    {174, 222, 190, 206, 195, 201, 200, 180},
    {217, 216, 199, 0, 203, 191, 217, 207},
    {175, 177, 206, 187, 241, 182, 186, 192},
    {223, 209, 213, 200, 225, 175, 174, 194},
    {178, 216, 225, 206, 180, 203, 211, 205},
    {203, 180, 220, 180, 196, 202, 216, 215},
    {191, 168, 187, 227, 211, 228, 230, 175},
    {213, 229, 179, 206, 177, 206, 220, 210},
    {198, 225, 182, 212, 179, 174, 197, 205},
    {170, 201, 228, 230, 180, 215, 185, 201},
    {185, 208, 187, 217, 227, 222, 181, 186},
    {185, 193, 171, 185, 204, 172, 206, 229},
    {205, 233, 218, 185, 214, 185, 184, 199},
    {225, 213, 185, 167, 226, 184, 200, 189},
    {199, 206, 181, 199, 179, 227, 173, 183},
    {214, 211, 194, 255, 166, 178, 220, 179}
};

static const uint8_t b2[8] = {199, 200, 199, 192, 205, 200, 201, 201};

static const uint8_t w3[8][1] = {
    {235},
    {240},
    {206},
    {162},
    {162},
    {185},
    {170},
    {225}
};

static const uint8_t b3[1] = {201};

    // Input Layer 
    float z1[16];
    for (int i = 0; i < 16; i++)
    {
        z1[i] = dequantized(b1[i]);
        for (int j = 0; j < 3; j++)
        {
            z1[i] += x[j] * dequantized(w1[j][i]);
        }
        z1[i] = relu(z1[i]);
    }

    // Hidden Layer 2
    float z2[8];
    for (int i = 0; i < 8; i++)
    {
        z2[i] = dequantized(b2[i]);
        for (int j = 0; j < 16; j++)
        {
            z2[i] += z1[j] * dequantized(w2[j][i]);
        }
        z2[i] = relu(z2[i]);
    }

    // Output Layer
    float z3[1];
    for (int k = 0; k < 1; k++)
    {
        z3[k] = dequantized(b3[k]);
        for (int j = 0; j < 8; j++)
        {
            z3[k] += z2[j] * dequantized(w3[j][k]);
        }
        y_pred[k] = linear(z3[k]);
    }

    return y_pred;
}

void free_prediction(float* prediction) {
    delete[] prediction;
}

protected:
float dequantized(uint8_t x)
{
    return (((x) / 255.0) * (0.9117356538772583 - -3.400634765625) + -3.400634765625);
};

float linear(float x)
{
    return x;
};

float relu(float x)
{
    return x > 0 ? x : 0;
};

};
}
}
