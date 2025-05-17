
pub struct MLP {
    input_layer: [f64; 784],
    hidden_layer_0: [f64; 16],
    hidden_layer_1: [f64; 16],
    output_layer: [f64; 10],
    weights_matrix: [[f64; 784]; 16]
}

impl MLP {

    pub fn new() -> Self {
        MLP {
            input_layer: [0.0; 784],
            hidden_layer_0: [0.0; 16],
            hidden_layer_1: [0.0; 16],
            output_layer: [0.0; 10],
            weights_matrix: [[0.0; 784]; 16],
        }
    }

    pub fn load_weights(&self) {
        println!("Hello YoYolops")
    }
}