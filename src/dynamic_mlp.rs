use rand_distr::{Distribution, Normal};
use nalgebra::{DMatrix, DVector};
use std::fs::File;
use std::io::{Read, Write, Result as IoResult};

use crate::constants::{INPUT_SIZE};

pub struct DMLP {
    layers: Vec<usize>,
    activations: Vec<DVector<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DVector<f64>>,
}

impl DMLP {
    pub fn new(layers: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();

        if layers.len() >= 2 {
            for i in 0..layers.len() - 1 {
                let fan_in = layers[i] as f64;
                let fan_out = layers[i + 1];
                let he_std = (2.0 / fan_in).sqrt();
                let dist = Normal::new(0.0, he_std).unwrap();

                let weight_matrix = DMatrix::from_fn(fan_out, layers[i], |_, _| dist.sample(&mut rng));
                let bias_vector = DVector::from_element(fan_out, 0.0);

                weights.push(weight_matrix);
                biases.push(bias_vector);
            }
        }

        for &layer_size in &layers {
            activations.push(DVector::from_element(layer_size, 0.0));
        }

        DMLP { layers, weights, biases, activations }
    }

    pub fn predict(&mut self, input: [f64; INPUT_SIZE]) -> DVector<f64> {
        if INPUT_SIZE != self.layers[0] {
            panic!(
                "DMLP configuration mismatch: 'predict' function compiled for INPUT_SIZE {} but DMLP instance is configured for input layer size {}.",
                INPUT_SIZE, self.layers[0]
            );
        }
        
        self.activations[0] = DVector::from_row_slice(&input);

        if self.weights.is_empty() {
             return self.activations[0].clone();
        }

        for i in 0..self.weights.len() {
            if (i + 1) >= self.activations.len() {
                panic!("DMLP internal inconsistency: insufficient activation vectors for layers/weights.");
            }

            let z = &self.weights[i] * &self.activations[i] + &self.biases[i];
            self.activations[i + 1] = if i == self.weights.len() - 1 {
                DMLP::softmax(&z)
            } else {
                DMLP::relu(&z)
            };
        }

        self.activations.last().expect("Activations vector should not be empty here").clone()
    }


    pub fn train_cross_entropy_batch(
        &mut self,
        images: &[Vec<f64>],
        labels: &[u8],
        learning_rate: f64,
    ) -> Vec<DVector<f64>> {
        if self.weights.is_empty() {
            eprintln!("Cannot train a network with no weights (e.g., less than 2 layers).");
            return Vec::new();
        }
        let batch_size = images.len();
        if batch_size == 0 {
            return Vec::new();
        }
        let mut predictions = Vec::with_capacity(batch_size);

        let mut grad_w_accum: Vec<DMatrix<f64>> = self.weights.iter()
            .map(|w| DMatrix::zeros(w.nrows(), w.ncols())).collect();
        let mut grad_b_accum: Vec<DVector<f64>> = self.biases.iter()
            .map(|b| DVector::zeros(b.len())).collect();

        for (x_vec, &label) in images.iter().zip(labels.iter()) { // x_vec is &Vec<f64>
            if x_vec.len() != INPUT_SIZE {
                panic!(
                    "Training data item size mismatch. Expected data items of size {} (INPUT_SIZE), but received an item of size {}.",
                    INPUT_SIZE,
                    x_vec.len()
                );
            }
            
            let mut image_array = [0.0f64; INPUT_SIZE]; // Create a fixed-size array
            image_array.copy_from_slice(x_vec.as_slice()); // Copy data from the Vec's slice

            let y_hat = self.predict(image_array); // Call predict with the array (passed by value)
            predictions.push(y_hat.clone());

            let mut target = DVector::from_element(self.layers.last().unwrap_or(&0).clone(), 0.0);
            if (label as usize) < target.len() {
                target[label as usize] = 1.0;
            } else {
                eprintln!("Warning: Label index {} is out of bounds for output layer size {}.", label, target.len());
                continue;
            }

            let mut delta = y_hat - target;

            for l in (0..self.weights.len()).rev() {
                grad_w_accum[l] += &delta * self.activations[l].transpose();
                grad_b_accum[l] += &delta;

                if l > 0 {
                    let mut new_delta = self.weights[l].transpose() * &delta;
                    for i in 0..new_delta.len() {
                        if self.activations[l][i] <= 0.0 {
                            new_delta[i] = 0.0;
                        }
                    }
                    delta = new_delta;
                }
            }
        }

        let learning_rate_avg = learning_rate / batch_size as f64;
        for i in 0..self.weights.len() {
            self.weights[i] -= &grad_w_accum[i] * learning_rate_avg;
            self.biases[i] -= &grad_b_accum[i] * learning_rate_avg;
        }
        predictions
    }

    pub fn show_weights(&self) {
        if self.weights.is_empty() {
            println!("The network has no weights (e.g., less than 2 layers defined).");
            return;
        }
        for (i, w) in self.weights.iter().enumerate() {
            println!("Weights between layer {} (size {}) and layer {} (size {}):", i, self.layers[i], i + 1, self.layers[i+1]);
            println!("{}", w);
        }
    }

    fn relu(x: &DVector<f64>) -> DVector<f64> {
        x.map(|v| v.max(0.0))
    }

    fn softmax(x: &DVector<f64>) -> DVector<f64> {
        if x.is_empty() {
            return DVector::from_vec(vec![]);
        }
        let max_val = x.max();
        let exps = x.map(|v| (v - max_val).exp());
        let sum: f64 = exps.sum();
        if sum == 0.0 {
            exps.map(|_| 1.0 / exps.len() as f64)
        } else {
            exps / sum
        }
    }

    pub fn show_layers(&self) {
        print!("[");
        for (i, &size) in self.layers.iter().enumerate() {
            print!("{}", size);
            if i < self.layers.len() - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    pub fn save_weights(&self, path: &str) -> IoResult<()> {
        let mut file = File::create(path)?;
        let num_layers_u32 = self.layers.len() as u32;
        file.write_all(&num_layers_u32.to_le_bytes())?;
        for &size in &self.layers {
            let size_u32 = size as u32;
            file.write_all(&size_u32.to_le_bytes())?;
        }
        for matrix in &self.weights {
            for &float_val in matrix.as_slice().iter() {
                file.write_all(&float_val.to_le_bytes())?;
            }
        }
        for vector in &self.biases {
            for &float_val in vector.as_slice().iter() {
                file.write_all(&float_val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load_weights(&mut self, path: &str) -> IoResult<()> {
        let mut file = File::open(path)?;
        let mut u32_buf = [0u8; 4];
        let mut f64_buf = [0u8; 8];

        file.read_exact(&mut u32_buf)?;
        let num_defined_layers = u32::from_le_bytes(u32_buf) as usize;

        let mut loaded_layers_vec = Vec::with_capacity(num_defined_layers);
        for _ in 0..num_defined_layers {
            file.read_exact(&mut u32_buf)?;
            let layer_size = u32::from_le_bytes(u32_buf) as usize;
            loaded_layers_vec.push(layer_size);
        }
        self.layers = loaded_layers_vec;

        self.weights.clear();
        self.biases.clear();

        if self.layers.len() >= 2 {
            for i in 0..(self.layers.len() - 1) {
                let rows = self.layers[i + 1];
                let cols = self.layers[i];
                let mut weight_matrix = DMatrix::<f64>::zeros(rows, cols);
                if rows * cols > 0 {
                    for val_ref in weight_matrix.as_mut_slice().iter_mut() {
                        file.read_exact(&mut f64_buf)?;
                        *val_ref = f64::from_le_bytes(f64_buf);
                    }
                }
                self.weights.push(weight_matrix);
            }
        }

        if self.layers.len() >= 2 {
            for i in 0..(self.layers.len() - 1) {
                let size = self.layers[i + 1];
                let mut bias_vector = DVector::<f64>::zeros(size);
                if size > 0 {
                    for val_ref in bias_vector.as_mut_slice().iter_mut() {
                        file.read_exact(&mut f64_buf)?;
                        *val_ref = f64::from_le_bytes(f64_buf);
                    }
                }
                self.biases.push(bias_vector);
            }
        }

        self.activations.clear();
        for &layer_size in &self.layers {
            self.activations.push(DVector::from_element(layer_size, 0.0));
        }
        Ok(())
    }

    pub fn render_output(&self, output_vector: &DVector<f64>) {
        const MAX_BAR_LENGTH: u32 = 50; // Maximum number of '█' characters per bar
    
        for (i, val) in output_vector.iter().enumerate() {
            let bar_len = (val * MAX_BAR_LENGTH as f64).round() as usize;
            let bar = "█".repeat(bar_len);
            println!("{:>2} | {:<width$} | {:.4}", i, bar, val, width = MAX_BAR_LENGTH as usize);
        }
        println!();
        println!("===========================================================");
        println!();
    }
}