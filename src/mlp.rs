use rand::Rng;
use nalgebra::{DMatrix, DVector};

pub struct MLP {
    input_layer: DVector<f64>,         // 784 x 1
    hidden_layer_0: DVector<f64>,      // 16 x 1
    hidden_layer_1: DVector<f64>,      // 16 x 1
    output_layer: DVector<f64>,        // 10 x 1
    weights_matrix_01: DMatrix<f64>,   // 16 x 784
    weights_matrix_12: DMatrix<f64>,   // 16 x 16
    weights_matrix_23: DMatrix<f64>,   // 10 x 16
}

impl MLP {

    pub fn new() -> Self {
        MLP {
            input_layer: DVector::from_element(784, 0.0),
            hidden_layer_0: DVector::from_element(16, 0.0),
            hidden_layer_1: DVector::from_element(16, 0.0),
            output_layer: DVector::from_element(10, 0.0),
            weights_matrix_01: DMatrix::from_element(16, 784, 0.0),
            weights_matrix_12: DMatrix::from_element(16, 16, 0.0),
            weights_matrix_23: DMatrix::from_element(10, 16, 0.0),
        }
    }

    pub fn randomize_weights(&mut self) {
        let mut rng = rand::rng();

        // Iterate through each weight matrix and assign random values
        for i in 0..self.weights_matrix_01.nrows() {
            for j in 0..self.weights_matrix_01.ncols() {
                self.weights_matrix_01[(i, j)] = rng.random_range(-1.0..1.0);
            }
        }

        for i in 0..self.weights_matrix_12.nrows() {
            for j in 0..self.weights_matrix_12.ncols() {
                self.weights_matrix_12[(i, j)] = rng.random_range(-1.0..1.0);
            }
        }

        for i in 0..self.weights_matrix_23.nrows() {
            for j in 0..self.weights_matrix_23.ncols() {
                self.weights_matrix_23[(i, j)] = rng.random_range(-1.0..1.0);
            }
        }
    }

    pub fn show_weights(&self) {
        println!("Weights Between Layers 0 & 1:");
        for row in self.weights_matrix_01.row_iter() {
            for val in row.iter() {
                print!("{:>6.1}", val);
            }
            println!();
        }

        println!("Weights Between Layers 1 & 2:");
        for row in self.weights_matrix_12.row_iter() {
            for val in row.iter() {
                print!("{:>6.1}", val);
            }
            println!();
        }

        println!("Weights Between Layers 2 & 3:");
        for row in self.weights_matrix_23.row_iter() {
            for val in row.iter() {
                print!("{:>6.1}", val);
            }
            println!();
        }
    }

    pub fn predict(&mut self, image: [f64; 784]) {
        // Fill the first layer with image data
        // self.input_layer = image;

        // multiply input layer vector by 01 weights matrix

        // 
    }

    pub fn load_weights(&self) {
        println!("Hello YoYolops, you didn't implemented this one yet :)");
    }
}