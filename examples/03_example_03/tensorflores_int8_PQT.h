namespace Conect2AI {
namespace TensorFlores {
class MultilayerPerceptron {
public: 

float* predict(float *x) { 
    float* y_pred = new float[1];
static const uint8_t w1[3][16] = {
    {118, 98, 66, 43, 165, 146, 99, 141, 119, 135, 87, 94, 161, 79, 100, 131},
    {46, 84, 129, 111, 107, 131, 189, 195, 120, 95, 156, 55, 155, 138, 83, 118},
    {148, 52, 164, 149, 139, 156, 113, 155, 78, 159, 50, 46, 146, 127, 139, 118}
};

static const uint8_t b1[16] = {118, 118, 119, 117, 119, 119, 119, 119, 119, 119, 119, 119, 118, 119, 117, 119};

static const uint8_t w2[16][8] = {
    {158, 110, 127, 130, 19, 166, 140, 154},
    {115, 115, 129, 89, 152, 114, 187, 168},
    {100, 133, 143, 92, 65, 141, 46, 71},
    {101, 131, 47, 141, 133, 182, 119, 131},
    {166, 135, 155, 132, 136, 231, 130, 159},
    {113, 147, 97, 142, 111, 86, 137, 102},
    {105, 118, 133, 166, 90, 24, 92, 110},
    {147, 154, 86, 99, 145, 187, 120, 167},
    {105, 87, 147, 128, 108, 147, 93, 120},
    {150, 84, 102, 151, 162, 69, 153, 131},
    {151, 66, 111, 106, 100, 139, 106, 48},
    {255, 80, 94, 116, 148, 138, 68, 42},
    {94, 155, 18, 113, 111, 29, 46, 129},
    {94, 67, 92, 131, 167, 153, 127, 85},
    {117, 60, 132, 149, 85, 32, 165, 119},
    {102, 94, 78, 158, 83, 147, 63, 63}
};

static const uint8_t b2[8] = {119, 119, 118, 119, 119, 118, 118, 118};

static const uint8_t w3[8][1] = {
    {137},
    {134},
    {84},
    {0},
    {85},
    {164},
    {85},
    {149}
};

static const uint8_t b3[1] = {118};

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
    return (((x) / 255.0) * (3.4116644580675497 - -2.9839453135012084) + -2.9839453135012084);
};

float relu(float x)
{
    return x > 0 ? x : 0;
};

float linear(float x)
{
    return x;
};

};
}
}
