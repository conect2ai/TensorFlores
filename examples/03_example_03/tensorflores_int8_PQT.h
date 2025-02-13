namespace Conect2AI {
namespace TensorFlores {
class MultilayerPerceptron {
public: 

float predict(float *x) { 
float y_pred = 0;
static const uint8_t w1[3][16] = {
    {146, 76, 120, 176, 207, 205, 255, 124, 204, 133, 0, 108, 158, 137, 74, 146},
    {170, 123, 133, 206, 87, 188, 95, 110, 136, 197, 87, 144, 81, 193, 97, 146},
    {189, 135, 102, 135, 191, 54, 194, 175, 110, 163, 231, 114, 106, 100, 74, 86}
};

static const uint8_t b1[16] = {134, 134, 135, 134, 134, 134, 133, 134, 134, 134, 135, 134, 134, 135, 134, 134};

static const uint8_t w2[16][8] = {
    {127, 152, 143, 109, 149, 188, 155, 183},
    {195, 206, 112, 185, 113, 236, 160, 146},
    {89, 102, 159, 171, 164, 89, 81, 167},
    {160, 48, 108, 189, 152, 140, 110, 86},
    {113, 110, 161, 168, 118, 108, 78, 89},
    {220, 99, 57, 150, 123, 199, 209, 200},
    {104, 90, 125, 142, 106, 177, 128, 148},
    {114, 200, 183, 205, 130, 208, 150, 114},
    {146, 135, 167, 124, 134, 215, 132, 58},
    {176, 96, 131, 89, 169, 82, 139, 111},
    {205, 109, 124, 109, 97, 134, 138, 172},
    {151, 80, 197, 138, 141, 229, 112, 94},
    {124, 120, 167, 124, 163, 149, 191, 35},
    {150, 142, 214, 198, 112, 61, 211, 189},
    {108, 81, 153, 227, 192, 146, 166, 56},
    {133, 160, 67, 240, 118, 97, 177, 63}
};

static const uint8_t b2[8] = {135, 134, 134, 134, 134, 135, 134, 134};

static const uint8_t w3[8][1] = {
    {158},
    {140},
    {139},
    {177},
    {107},
    {115},
    {84},
    {144}
};

static const uint8_t b3[1] = {134};

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
    float z3 = dequantized(b3[0]);
    for (int i = 0; i < 8; i++)
    {
        z3 += z2[i] * dequantized(w3[i][0]);
        z3 = linear(z3);
    }

y_pred = z3;
return y_pred;
}
protected:
float dequantized(uint8_t x)
{
    return (((x) / 255.0) * (2.485611518934096 - -2.7866812544379806) + -2.7866812544379806);
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
