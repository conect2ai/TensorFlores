namespace Conect2AI
{
    namespace TensorFlores
    {
        class MultilayerPerceptron
        {
        public:
            float predict(float *x)
            {
                float y_pred = 0;
                static const float center_bias[3] = {-0.003076801320526642, 0.016718890732474484, -0.012545546282279465};

                static const float centers_weights[9] = {-0.03278433166048851, -0.9289895372711547, -1.940861781673081, -1.434814908572797, 1.9116050989404338, 1.1623880597123224, -0.8888332538018183, -1.2794115454669592, 0.7939062206044778};

                static const uint8_t w1[3][16] = {
                    {0, 6, 6, 0, 3, 6, 1, 1, 0, 1, 0, 0, 7, 8, 8, 0},
                    {6, 2, 0, 3, 6, 0, 0, 2, 0, 1, 6, 0, 0, 0, 0, 8},
                    {0, 8, 0, 6, 1, 1, 2, 1, 1, 2, 2, 0, 0, 6, 2, 2}};

                static const uint8_t b1[16] = {2, 2, 2, 1, 0, 0, 1, 1, 2, 1, 0, 2, 0, 0, 2, 1};

                static const uint8_t w2[16][8] = {
                    {6, 5, 2, 8, 6, 0, 8, 8},
                    {0, 7, 2, 0, 8, 6, 1, 0},
                    {8, 6, 6, 8, 6, 1, 6, 6},
                    {0, 3, 3, 4, 6, 0, 6, 7},
                    {6, 6, 0, 8, 8, 8, 0, 0},
                    {6, 6, 8, 0, 2, 6, 0, 0},
                    {1, 6, 3, 3, 6, 2, 1, 3},
                    {8, 2, 2, 4, 0, 0, 0, 5},
                    {6, 0, 6, 5, 6, 0, 4, 4},
                    {3, 1, 0, 8, 8, 6, 0, 0},
                    {0, 6, 8, 4, 8, 4, 0, 6},
                    {1, 0, 4, 4, 0, 0, 8, 6},
                    {0, 0, 8, 0, 4, 6, 7, 7},
                    {6, 8, 6, 8, 6, 5, 0, 0},
                    {2, 6, 0, 0, 1, 6, 8, 8},
                    {1, 0, 0, 2, 5, 8, 0, 0}};

                static const uint8_t b2[8] = {0, 0, 0, 1, 2, 0, 2, 0};

                static const uint8_t w3[8][1] = {
                    {6},
                    {8},
                    {0},
                    {8},
                    {0},
                    {0},
                    {6},
                    {8}};

                static const uint8_t b3[1] = {2};

                // Camada de Entrada
                float z1[16];
                for (int i = 0; i < 16; i++)
                {
                    z1[i] = center_bias[b1[i]];
                    for (int j = 0; j < 3; j++)
                    {
                        z1[i] += x[j] * centers_weights[w1[j][i]];
                    }
                    z1[i] = relu(z1[i]);
                }

                // Camada Oculta 2
                float z2[8];
                for (int i = 0; i < 8; i++)
                {
                    z2[i] = center_bias[b2[i]];
                    for (int j = 0; j < 16; j++)
                    {
                        z2[i] += z1[j] * centers_weights[w2[j][i]];
                    }
                    z2[i] = relu(z2[i]);
                }

                // Camada de Sa�da
                float z3 = center_bias[b3[0]];
                for (int i = 0; i < 8; i++)
                {
                    z3 += z2[i] * centers_weights[w3[i][0]];
                    z3 = linear(z3);
                }

                y_pred = z3;
                return y_pred;
            }

        protected:
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
