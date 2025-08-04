namespace Conect2AI {
namespace TensorFlores {
class MultilayerPerceptron {
public: 

float* predict(float *x) {
    float* y_pred = new float[1];
static const float center_bias[3] = {-0.005864434900393029, 0.009274461857028998, -0.01754724964363693};

static const float centers_weights[6] = {0.23761001883118163, 1.129387859258356, -0.5437767693093417, -2.387265721871522, -2.2969898859883666, 1.1835949048933188};

static const uint8_t w1[3][16] = {
    {0, 5, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 5, 4},
    {0, 0, 4, 4, 0, 3, 2, 2, 2, 4, 2, 2, 3, 4, 4, 2},
    {4, 5, 5, 2, 2, 4, 2, 2, 2, 5, 3, 4, 2, 2, 5, 0}
};

static const uint8_t b1[16] = {0, 0, 1, 1, 2, 1, 2, 0, 0, 1, 1, 0, 2, 2, 1, 0};

static const uint8_t w2[16][8] = {
    {0, 1, 2, 0, 0, 2, 2, 0},
    {3, 4, 2, 0, 3, 2, 4, 3},
    {2, 4, 2, 0, 0, 0, 2, 2},
    {0, 0, 1, 2, 2, 0, 2, 2},
    {0, 5, 0, 2, 2, 1, 5, 0},
    {5, 2, 0, 2, 5, 5, 0, 2},
    {3, 0, 2, 1, 0, 2, 2, 2},
    {2, 2, 4, 0, 2, 4, 0, 5},
    {4, 0, 0, 1, 1, 0, 0, 2},
    {2, 2, 0, 0, 2, 0, 4, 2},
    {0, 0, 0, 1, 2, 2, 0, 5},
    {0, 2, 0, 1, 0, 0, 4, 2},
    {4, 2, 1, 2, 5, 2, 0, 5},
    {0, 4, 2, 5, 2, 5, 0, 2},
    {0, 2, 0, 2, 2, 0, 0, 4},
    {0, 1, 2, 2, 0, 2, 2, 0}
};

static const uint8_t b2[8] = {1, 0, 0, 0, 2, 2, 1, 1};

static const uint8_t w3[8][1] = {
    {2},
    {2},
    {0},
    {1},
    {2},
    {0},
    {0},
    {3}
};

static const uint8_t b3[1] = {0};

    // Input Layer 
    float z1[16];
    for (int i = 0; i < 16; i++)
    {
        z1[i] = center_bias[b1[i]];
        for (int j = 0; j < 3; j++)
        {
            z1[i] += x[j] * centers_weights[w1[j][i]];
        }
        z1[i] = relu(z1[i]);    }

    // Hidden Layers 
    float z2[8];
    for (int i = 0; i < 8; i++)
    {
        z2[i] = center_bias[b2[i]];
        for (int j = 0; j < 16; j++)
        {
            z2[i] += z1[j] * centers_weights[w2[j][i]];
        }
        z2[i] = relu(z2[i]);    }

    // Output Layer
    float z3[1];
    for (int k = 0; k < 1; k++)
    {
        z3[k] = center_bias[b3[k]];
        for (int j = 0; j < 8; j++)
        {
            z3[k] += z2[j] * centers_weights[w3[j][k]];
        }
        y_pred[k] = linear(z3[k]);    }

    return y_pred;
}
void free_prediction(float* prediction) {
    delete[] prediction;
}

protected:
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
