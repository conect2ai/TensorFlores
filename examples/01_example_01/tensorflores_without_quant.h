namespace Conect2AI {
namespace TensorFlores {
class MultilayerPerceptron {
public: 

float predict(float *x) { 
float y_pred = 0;
static const float w1[3][16] = {
    {0.25130431662396613, -1.2147914404241322, -0.2884391170139249, 0.8606677711806703, 1.5018800063241295, 1.457068347654409, 2.485611518934096, -0.2194694221104431, 1.44672478778409, -0.036373508158682274, -2.7866812544379806, -0.5462811108638331, 0.5005394347543287, 0.046451653299363044, -1.241076380766991, 0.24541163169106964},
    {0.7360947147160628, -0.22909043202182777, -0.02888562950803397, 1.4740145700748322, -0.972825532772787, 1.1031430486796638, -0.8224112838985033, -0.4990274814217257, 0.038942922605802455, 1.3026129213060025, -0.977868792905857, 0.19823389720251328, -1.0935966078970152, 1.2092028471045888, -0.7700673867395008, 0.24021322746849333},
    {1.1378215866161998, 0.008643173246605705, -0.6594804193082676, 0.016496096434585293, 1.1809950202488912, -1.6618199459360117, 1.2391240109167398, 0.8348225347407953, -0.4982755277223971, 0.5887411396126568, 1.9910831205064137, -0.410360722221189, -0.5844452903962888, -0.707437360100421, -1.2415350115018677, -1.0085503978733108}
};

static const float b1[16] = {-0.00661694117823328, -0.011793890873857609, 0.008761852875022819, -0.012785585881086903, -0.0027887780188731335, 0.0005198726109704008, -0.01640923879334639, -0.0040365228696589706, -0.004333346605972805, -0.006451286568398461, 0.02048311797735747, -0.014754874098237693, -0.010347294840309273, 0.006467824805941737, 0.002257082829689511, -0.0007130371317065206};

static const float w2[16][8] = {
    {-0.1485636793511766, 0.3569973851800046, 0.17908540512565607, -0.523877698898761, 0.2945924250968459, 1.116635008713201, 0.42687132679557116, 1.0102292672725561},
    {1.2547210736366834, 1.4752272780003175, -0.45328894351466564, 1.0414994898562566, -0.4472473205849872, 2.0979128794616053, 0.5269527636182276, 0.2518133406709035},
    {-0.9441392957123559, -0.6750334563431777, 0.501346465811451, 0.7561621982059055, 0.6152131928814151, -0.9420826372525032, -1.1021618286196757, 0.6699094525939754},
    {0.5352327723617429, -1.7938438142587556, -0.5343812365606473, 1.1396737616705106, 0.36302727001850404, 0.11457044902237028, -0.5067370410922932, -0.9955378494486228},
    {-0.4478905378279604, -0.5032414043436796, 0.5580158757981062, 0.6985938355761102, -0.3293936902581421, -0.5377471148046655, -1.1656855330407871, -0.9418728454323618},
    {1.763680904867001, -0.7304675691115415, -1.5941530682113298, 0.32190105564576454, -0.2341177492162813, 1.343946796665376, 1.5432795962406816, 1.3548460259580757},
    {-0.6158331790837044, -0.9075132587846512, -0.18737321800424672, 0.16804536046932989, -0.5872912647331963, 0.8814396473790074, -0.13163806140116968, 0.28043119455991633},
    {-0.418018042618772, 1.3550861118141613, 1.0077904811716305, 1.4557116683102709, -0.08877165232197029, 1.532732528696962, 0.32151601545443453, -0.4252916316223109},
    {0.2355449822832505, 0.021011609020350205, 0.6765907336904919, -0.2055125703299116, -0.012019329622008641, 1.6773548969042094, -0.05185310700886722, -1.5774350462048383},
    {0.8724162811956299, -0.7925989074620606, -0.07749276692811417, -0.9309781561471873, 0.7198293085493946, -1.0770913258756176, 0.09791018116772175, -0.48234644852580866},
    {1.4647440293416336, -0.5298677585198952, -0.20334955368608285, -0.5125437835940555, -0.7764562281027877, 0.004242222388113183, 0.08571038693364064, 0.7869639011680545},
    {0.34945572469841635, -1.1314617429752925, 1.2865080774841469, 0.06977886431710244, 0.13661799544935907, 1.9597726048337496, -0.46538064813283053, -0.8371807153209272},
    {-0.22277421167151476, -0.2862962272219545, 0.6810733009125609, -0.2045169607471015, 0.5905001708630732, 0.312131402668764, 1.1670274243342988, -2.047408478142414},
    {0.3274117151897499, 0.16215166284921767, 1.63865111365463, 1.3079013742172985, -0.4613366756451735, -1.5217601730486077, 1.5827349351697597, 1.140247155803638},
    {-0.5495862537280967, -1.1010219911121708, 0.3912969124561298, 1.9191618950105565, 1.1948614254162662, 0.2330976392748248, 0.6599943704106348, -1.6211173406238175},
    {-0.03489815422666802, 0.5227145409950461, -1.383323314036505, 2.1831090028509856, -0.3359312163901842, -0.7739581288762516, 0.8886294000043664, -1.467504544906624}
};

static const float b2[8] = {0.00614338070753355, -0.0013846056296751823, -0.00470371281584063, -0.0008368695999715229, 0.0002615576281515959, 0.010889836801382757, 0.003939469766272716, 0.00184923021499863};

static const float w3[8][1] = {
    {0.4824099067055086},
    {0.1100612727241187},
    {0.0986326662007946},
    {0.8838276773487592},
    {-0.5556334860221619},
    {-0.407548802718415},
    {-1.0388067433395756},
    {0.20052746747409889}
};

static const float b3[1] = {0.0034703559530694685};

    // Camada de Entrada 
    float z1[16];
    for (int i = 0; i < 16; i++)
    {
        z1[i] = b1[i];
        for (int j = 0; j < 3; j++)
        {
            z1[i] += x[j] * w1[j][i];
        }
        z1[i] = relu(z1[i]);    }

    // Camada Oculta 2
    float z2[8];
    for (int i = 0; i < 8; i++)
    {
        z2[i] = b2[i];
        for (int j = 0; j < 16; j++)
        {
            z2[i] += z1[j] * w2[j][i];
        }
        z2[i] = relu(z2[i]);    }

    // Camada de Sa�da
    float z3 = b3[0];
    for (int i = 0; i < 8; i++)
     {
       z3 += z2[i] * w3[i][0];
       z3 = linear(z3);     }

y_pred = z3;
return y_pred;}
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
