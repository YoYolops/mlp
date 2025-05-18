use rand::Rng;

pub struct MLP {
    input_layer: [f64; 784],
    hidden_layer_0: [f64; 16],
    hidden_layer_1: [f64; 16],
    output_layer: [f64; 10],
    weights_matrix_01: [[f64; 784]; 16],
    weights_matrix_12: [[f64; 16]; 16],
    weights_matrix_23: [[f64; 16]; 10]
}

impl MLP {

    pub fn new() -> Self {
        MLP {
            input_layer: [0.0; 784],
            hidden_layer_0: [0.0; 16],
            hidden_layer_1: [0.0; 16],
            output_layer: [0.0; 10],
            weights_matrix_01: [[0.0; 784]; 16],
            weights_matrix_12: [[0.0; 16]; 16],
            weights_matrix_23: [[0.0; 16]; 10]
        }
    }

    pub fn randomize_weights(&mut self) {
        let mut rng = rand::rng();
        
        for row in self.weights_matrix_01.iter_mut() {
            for weight in row.iter_mut() {
                *weight = rng.random_range(-1.0..1.0);
            }
        }

        for row in self.weights_matrix_12.iter_mut() {
            for weight in row.iter_mut() {
                *weight = rng.random_range(-1.0..1.0);
            }
        }

        for row in self.weights_matrix_23.iter_mut() {
            for weight in row.iter_mut() {
                *weight = rng.random_range(-1.0..1.0);
            }
        }
    }

    pub fn show_weights(&self) {
        println!("Weights Between Layers 0 & 1:");
        for row in self.weights_matrix_01 {
            for val in row {
                print!("{:>4.1}", val);
            }
            println!();
        }

        println!("Weights Between Layers 1 & 2:");
        for row in self.weights_matrix_12 {
            for val in row {
                print!("{:>6.1}", val);
            }
            println!();
        }

        println!("Weights Between Layers 2 & 3:");
        for row in self.weights_matrix_23 {
            for val in row {
                print!("{:>6.1}", val);
            }
            println!();
        }
    }

    pub fn load_weights(&self) {
        println!("Hello YoYolops, you didn't implemented this one yet :)");
    }
}