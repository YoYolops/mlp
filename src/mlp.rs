use rand::rng;
use rand_distr::{Distribution, Normal};
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
    hidden_bias_0: SVector<f64, HIDDEN_SIZE>,
    hidden_bias_1: SVector<f64, HIDDEN_SIZE>,
    output_bias: SVector<f64, OUTPUT_SIZE>,
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
            hidden_bias_0: SVector::<f64, HIDDEN_SIZE>::from_element(0.0),
            hidden_bias_1: SVector::<f64, HIDDEN_SIZE>::from_element(0.0),
            output_bias: SVector::<f64, OUTPUT_SIZE>::from_element(0.0),
        }
    }

    pub fn randomize_weights(&mut self) {
        let mut rng = rng();

        // He initialization for weights_matrix_01 (fan_in = INPUT_SIZE)
        let he_std_01 = (2.0 / INPUT_SIZE as f64).sqrt();
        let dist_01 = Normal::new(0.0, he_std_01).unwrap();
        for i in 0..self.weights_matrix_01.nrows() {
            for j in 0..self.weights_matrix_01.ncols() {
                self.weights_matrix_01[(i, j)] = dist_01.sample(&mut rng);
            }
        }

        // He initialization for weights_matrix_12 (fan_in = HIDDEN_SIZE)
        let he_std_12 = (2.0 / HIDDEN_SIZE as f64).sqrt();
        let dist_12 = Normal::new(0.0, he_std_12).unwrap();
        for i in 0..self.weights_matrix_12.nrows() {
            for j in 0..self.weights_matrix_12.ncols() {
                self.weights_matrix_12[(i, j)] = dist_12.sample(&mut rng);
            }
        }

        // He initialization for weights_matrix_23 (fan_in = HIDDEN_SIZE)
        let he_std_23 = (2.0 / HIDDEN_SIZE as f64).sqrt();
        let dist_23 = Normal::new(0.0, he_std_23).unwrap();
        for i in 0..self.weights_matrix_23.nrows() {
            for j in 0..self.weights_matrix_23.ncols() {
                self.weights_matrix_23[(i, j)] = dist_23.sample(&mut rng);
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

    fn apply_relu<const N: usize>(layer: &mut SVector<f64, N>) {
        for val in layer.iter_mut() {
            *val = val.max(0.0);
        }
    }

    fn softmax<const N: usize>(layer: &SVector<f64, N>) -> SVector<f64, N> {
        let max_val = layer.max(); // for numerical stability
        let exps = layer.map(|x| (x - max_val).exp());
        let sum: f64 = exps.iter().sum();
        exps / sum
    }

    // It is good the image is not borrowed, since after prediction, the value can be dropped
    pub fn predict(&mut self, image: [f64; INPUT_SIZE]) -> SVector<f64, OUTPUT_SIZE> {
        self.input_layer = SVector::<f64, INPUT_SIZE>::from_row_slice(&image);

        self.hidden_layer_0 = (self.weights_matrix_01 * self.input_layer) + self.hidden_bias_0;
        MLP::apply_relu(&mut self.hidden_layer_0);

        self.hidden_layer_1 = (self.weights_matrix_12 * self.hidden_layer_0) + self.hidden_bias_1;
        MLP::apply_relu(&mut self.hidden_layer_1);

        self.output_layer = (self.weights_matrix_23 * self.hidden_layer_1) + self.output_bias;
        MLP::softmax(&self.output_layer)
    }

    pub fn train_cross_entropy(
        &mut self,
        image: [f64; INPUT_SIZE],
        target_label: u8,
        learning_rate: f64,
    ) -> SVector<f64, OUTPUT_SIZE> {
        // Forward pass
        let prediction = self.predict(image);

        // Create one-hot encoded target vector
        let mut target = SVector::<f64, OUTPUT_SIZE>::from_element(0.0);
        target[target_label as usize] = 1.0;

        // === Backpropagation ===

        // Gradient of Cross-Entropy Loss with Softmax:
        // grad = prediction - target
        let delta_output = prediction - target;

        // Gradient for weights_matrix_23 and output_bias
        let grad_w23 = delta_output * self.hidden_layer_1.transpose();
        let grad_b_output = delta_output;

        // Backprop to hidden_layer_1
        let mut delta_hidden_1 = self.weights_matrix_23.transpose() * delta_output;
        for i in 0..HIDDEN_SIZE {
            if self.hidden_layer_1[i] <= 0.0 {
                delta_hidden_1[i] = 0.0; // ReLU derivative
            }
        }

        // Gradient for weights_matrix_12 and bias
        let grad_w12 = delta_hidden_1 * self.hidden_layer_0.transpose();
        let grad_b_hidden_1 = delta_hidden_1;

        // Backprop to hidden_layer_0
        let mut delta_hidden_0 = self.weights_matrix_12.transpose() * delta_hidden_1;
        for i in 0..HIDDEN_SIZE {
            if self.hidden_layer_0[i] <= 0.0 {
                delta_hidden_0[i] = 0.0; // ReLU derivative
            }
        }

        let grad_w01 = delta_hidden_0 * self.input_layer.transpose();
        let grad_b_hidden_0 = delta_hidden_0;

        // === Weight & Bias Updates ===
        self.weights_matrix_23 -= learning_rate * grad_w23;
        self.output_bias -= learning_rate * grad_b_output;

        self.weights_matrix_12 -= learning_rate * grad_w12;
        self.hidden_bias_1 -= learning_rate * grad_b_hidden_1;

        self.weights_matrix_01 -= learning_rate * grad_w01;
        self.hidden_bias_0 -= learning_rate * grad_b_hidden_0;
        prediction
    }

    pub fn load_weights(&self) {
        println!("Hello YoYolops, you didn't implemented this one yet :)");
    }

    pub fn save_weights(&self) {
        println!("Unfortunately you also did not implement this one yet")
    }

}