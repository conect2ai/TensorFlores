#include "model.h"
#include "input_output.h"


float start_time = -1;
float end_time = -1;
float width_time = -1;

Conect2AI::TensorFlores::MultilayerPerceptron ANN;

void setup() {
  Serial.begin(9600);
}

void loop() {
  delay(2000);
  Serial.println("START");
  for (int i = 0; i < sizeof(output_data) / sizeof(output_data[0]); i++) {
    
    // Capture initial time
    start_time = micros();

    // Make a prediction
    float* y_pred = ANN.predict(input_data[i]);

    // Capture final time
    end_time = micros();
    width_time = end_time - start_time;

    // Display real output and prediction
    //Serial.print("Real: ");
    Serial.print(output_data[i], 8);
    //Serial.print(", Predicted: ");
    Serial.print(";");
    Serial.print(y_pred[0], 8);
    //Serial.print(", Time: ");
    Serial.print(";");
    Serial.println(width_time);
    //Serial.println(" us");
  }
  Serial.println("END");
  delay(20000);
}


