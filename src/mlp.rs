use rand::Rng;
use nalgebra::{SMatrix, SVector};
use crate::constants::{OUTPUT_SIZE, HIDDEN_SIZE, INPUT_SIZE};

pub struct MLP {
    input_layer: SVector<f64, INPUT_SIZE>,              // 784 x 1
    hidden_layer_0: SVector<f64, HIDDEN_SIZE>,          // 16 x 1
    hidden_layer_1: SVector<f64, HIDDEN_SIZE>,          // 16 x 1
    output_layer: SVector<f64, OUTPUT_SIZE>,            // 10 x 1
    weights_matrix_01: SMatrix<f64, HIDDEN_SIZE, INPUT_SIZE>, // 16 x 784
    weights_matrix_12: SMatrix<f64, HIDDEN_SIZE, HIDDEN_SIZE>, // 16 x 16
    weights_matrix_23: SMatrix<f64, OUTPUT_SIZE, HIDDEN_SIZE>, //   // 10 x 16
}

impl MLP {

    pub fn new() -> Self {
        MLP {
            input_layer: SVector::<f64, INPUT_SIZE>::from_element(0.0),
            hidden_layer_0: SVector::<f64, HIDDEN_SIZE>::from_element(0.0),
            hidden_layer_1: SVector::<f64, HIDDEN_SIZE>::from_element(0.0),
            output_layer: SVector::<f64, OUTPUT_SIZE>::from_element(0.0),
            weights_matrix_01: SMatrix::<f64, HIDDEN_SIZE, INPUT_SIZE>::from_element(0.0),
            weights_matrix_12: SMatrix::<f64, HIDDEN_SIZE, HIDDEN_SIZE>::from_element(0.0),
            weights_matrix_23: SMatrix::<f64, OUTPUT_SIZE, HIDDEN_SIZE>::from_element(0.0),
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


    fn apply_relu<const N: usize>(&self, layer: &mut SVector<f64, N>) {
        for val in layer.iter_mut() {
            *val = val.max(0.0);
        }
    }

    pub fn predict(&mut self, image: [f64; INPUT_SIZE]) {
        self.input_layer = SVector::<f64, INPUT_SIZE>::from_row_slice(&image);
        self.hidden_layer_0 = self.weights_matrix_01 * self.input_layer;
        // Relu must be applied in every layer after firing neurons
        self.apply_relu(&mut self.hidden_layer_0);  // <-- call on self & pass mutable ref
        self.hidden_layer_1 = self.weights_matrix_12 * self.hidden_layer_0;
        self.output_layer = self.weights_matrix_23 * self.hidden_layer_1;

        println!("{:?}", self.output_layer);
    }


    pub fn load_weights(&self) {
        println!("Hello YoYolops, you didn't implemented this one yet :)");
    }


}