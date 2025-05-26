use rand_distr::{Distribution, Normal};
use nalgebra::{DMatrix, DVector};
use std::fs::File;
use std::io::{Read, Write, Result as IoResult};

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

    pub fn predict(&mut self, input: &[f64]) -> DVector<f64> {
        if self.layers.is_empty() || self.activations.is_empty() {
            return DVector::from_vec(vec![]);
        }
        if input.len() != self.layers[0] {
            panic!("Input dimension mismatch. Expected {}, got {}", self.layers[0], input.len());
        }

        self.activations[0] = DVector::from_row_slice(input);

        if self.weights.is_empty() {
             return self.activations[0].clone();
        }

        for i in 0..self.weights.len() {
            let z = &self.weights[i] * &self.activations[i] + &self.biases[i];
            self.activations[i + 1] = if i == self.weights.len() - 1 {
                DMLP::softmax(&z)
            } else {
                DMLP::relu(&z)
            };
        }
        self.activations.last().unwrap().clone()
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

        for (x, &label) in images.iter().zip(labels.iter()) {
            let y_hat = self.predict(x);
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

    // Saves the DMLP's layers, weights, and biases to a binary file.
    pub fn save_weights(&self, path: &str) -> IoResult<()> {
        let mut file = File::create(path)?;

        // Save metadata:
        // number of layers
        let num_layers_u32 = self.layers.len() as u32;
        file.write_all(&num_layers_u32.to_le_bytes())?;

        // Layers sizes
        for &size in &self.layers {
            let size_u32 = size as u32;
            file.write_all(&size_u32.to_le_bytes())?;
        }

        // Weight Matrices
        for matrix in &self.weights {
            for &float_val in matrix.as_slice().iter() {
                file.write_all(&float_val.to_le_bytes())?;
            }
        }

        // Biases
        for vector in &self.biases {
            for &float_val in vector.as_slice().iter() {
                file.write_all(&float_val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    // Loads the DMLP's layers, weights, and biases from a binary file.
    // This will reconfigure the current DMLP instance.
    pub fn load_weights(&mut self, path: &str) -> IoResult<()> {
        let mut file = File::open(path)?;
        let mut u32_buf = [0u8; 4];
        let mut f64_buf = [0u8; 8];

        // Read number of layers
        file.read_exact(&mut u32_buf)?;
        let num_defined_layers = u32::from_le_bytes(u32_buf) as usize;

        // Read layers sizes
        let mut loaded_layers_vec = Vec::with_capacity(num_defined_layers);
        for _ in 0..num_defined_layers {
            file.read_exact(&mut u32_buf)?;
            let layer_size = u32::from_le_bytes(u32_buf) as usize;
            loaded_layers_vec.push(layer_size);
        }
        self.layers = loaded_layers_vec;

        self.weights.clear();
        self.biases.clear();

        // Load weight matrices
        if self.layers.len() >= 2 {
            for i in 0..(self.layers.len() - 1) {
                let rows = self.layers[i + 1];
                let cols = self.layers[i];
                let mut weight_matrix = DMatrix::<f64>::zeros(rows, cols);
                if rows * cols > 0 { // Only read if there are elements
                    for val_ref in weight_matrix.as_mut_slice().iter_mut() {
                        file.read_exact(&mut f64_buf)?;
                        *val_ref = f64::from_le_bytes(f64_buf);
                    }
                }
                self.weights.push(weight_matrix);
            }
        }

        // Load bias vectors
        if self.layers.len() >= 2 {
            for i in 0..(self.layers.len() - 1) {
                let size = self.layers[i + 1]; // Bias vector corresponds to the output neurons of the layer
                let mut bias_vector = DVector::<f64>::zeros(size);
                if size > 0 { // Only read if there are elements
                    for val_ref in bias_vector.as_mut_slice().iter_mut() {
                        file.read_exact(&mut f64_buf)?;
                        *val_ref = f64::from_le_bytes(f64_buf);
                    }
                }
                self.biases.push(bias_vector);
            }
        }

        // Re-initialize activations based on new layer structure
        self.activations.clear();
        for &layer_size in &self.layers {
            self.activations.push(DVector::from_element(layer_size, 0.0));
        }
        Ok(())
    }
}