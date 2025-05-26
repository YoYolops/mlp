use std::fs::File;
use std::io::{Read, Result, Write, ErrorKind};

use rand::rng; // Assuming this is intended, or use rand::thread_rng()
use rand_distr::{Distribution, Normal};
use nalgebra::{SMatrix, SVector};
use crate::constants::{OUTPUT_SIZE, HIDDEN_SIZE_0, HIDDEN_SIZE_1, INPUT_SIZE};

// Magic number is removed as per your request.

pub struct SMLP {
    input_layer: SVector<f64, INPUT_SIZE>,
    hidden_layer_0: SVector<f64, HIDDEN_SIZE_0>,
    hidden_layer_1: SVector<f64, HIDDEN_SIZE_1>,
    output_layer: SVector<f64, OUTPUT_SIZE>,
    weights_matrix_01: SMatrix<f64, HIDDEN_SIZE_0, INPUT_SIZE>,
    weights_matrix_12: SMatrix<f64, HIDDEN_SIZE_1, HIDDEN_SIZE_0>,
    weights_matrix_23: SMatrix<f64, OUTPUT_SIZE, HIDDEN_SIZE_1>,
    hidden_bias_0: SVector<f64, HIDDEN_SIZE_0>,
    hidden_bias_1: SVector<f64, HIDDEN_SIZE_1>,
    output_bias: SVector<f64, OUTPUT_SIZE>,
}

impl SMLP {
    pub fn new() -> Self {
        SMLP {
            input_layer: SVector::from_element(0.0),
            hidden_layer_0: SVector::from_element(0.0),
            hidden_layer_1: SVector::from_element(0.0),
            output_layer: SVector::from_element(0.0),
            weights_matrix_01: SMatrix::from_element(0.0),
            weights_matrix_12: SMatrix::from_element(0.0),
            weights_matrix_23: SMatrix::from_element(0.0),
            hidden_bias_0: SVector::from_element(0.0),
            hidden_bias_1: SVector::from_element(0.0),
            output_bias: SVector::from_element(0.0),
        }
    }

    pub fn randomize_weights(&mut self) {
        let mut rng = rand::thread_rng();

        let he_std_01 = (2.0 / INPUT_SIZE as f64).sqrt();
        let dist_01 = Normal::new(0.0, he_std_01).unwrap();
        self.weights_matrix_01.iter_mut().for_each(|x| *x = dist_01.sample(&mut rng));

        let he_std_12 = (2.0 / HIDDEN_SIZE_0 as f64).sqrt();
        let dist_12 = Normal::new(0.0, he_std_12).unwrap();
        self.weights_matrix_12.iter_mut().for_each(|x| *x = dist_12.sample(&mut rng));

        let he_std_23 = (2.0 / HIDDEN_SIZE_1 as f64).sqrt();
        let dist_23 = Normal::new(0.0, he_std_23).unwrap();
        self.weights_matrix_23.iter_mut().for_each(|x| *x = dist_23.sample(&mut rng));
    }

    fn get_params(&self) -> [&[f64]; 6] {
        [
            self.weights_matrix_01.as_slice(),
            self.weights_matrix_12.as_slice(),
            self.weights_matrix_23.as_slice(),
            self.hidden_bias_0.as_slice(),
            self.hidden_bias_1.as_slice(),
            self.output_bias.as_slice(),
        ]
    }

    fn get_params_mut(&mut self) -> [&mut [f64]; 6] {
        [
            self.weights_matrix_01.as_mut_slice(),
            self.weights_matrix_12.as_mut_slice(),
            self.weights_matrix_23.as_mut_slice(),
            self.hidden_bias_0.as_mut_slice(),
            self.hidden_bias_1.as_mut_slice(),
            self.output_bias.as_mut_slice(),
        ]
    }

    /// Saves the SMLP's structure metadata (dimensions) and weights/biases to a binary file.
    /// All numeric data is stored in little-endian format.
    pub fn save_weights(&self, path: &str) -> Result<()> {
        let mut file = File::create(path)?;

        // 1. Write structural constants (as u32 little-endian)
        file.write_all(&(INPUT_SIZE as u32).to_le_bytes())?;
        file.write_all(&(HIDDEN_SIZE_0 as u32).to_le_bytes())?;
        file.write_all(&(HIDDEN_SIZE_1 as u32).to_le_bytes())?;
        file.write_all(&(OUTPUT_SIZE as u32).to_le_bytes())?;

        // 2. Write parameters (f64 values as little-endian)
        for param_slice in self.get_params() {
            for &float_val in param_slice.iter() {
                file.write_all(&float_val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Loads the SMLP's weights and biases from a binary file.
    /// Verifies structure metadata (dimensions) before loading. Assumes little-endian format.
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        let mut file = File::open(path)?;
        let mut u32_buffer = [0u8; 4];
        let mut f64_buffer = [0u8; 8];

        // 1. Read structural constants from file
        file.read_exact(&mut u32_buffer)?;
        let file_input_size = u32::from_le_bytes(u32_buffer);
        file.read_exact(&mut u32_buffer)?;
        let file_hidden_size_0 = u32::from_le_bytes(u32_buffer);
        file.read_exact(&mut u32_buffer)?;
        let file_hidden_size_1 = u32::from_le_bytes(u32_buffer);
        file.read_exact(&mut u32_buffer)?;
        let file_output_size = u32::from_le_bytes(u32_buffer);

        // 2. Compare with current SMLP's compile-time constants
        if file_input_size != INPUT_SIZE as u32 ||
           file_hidden_size_0 != HIDDEN_SIZE_0 as u32 ||
           file_hidden_size_1 != HIDDEN_SIZE_1 as u32 ||
           file_output_size != OUTPUT_SIZE as u32 {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "SMLP structure mismatch. Expected (I,H0,H1,O): [{},{},{},{}], file has: [{},{},{},{}]",
                    INPUT_SIZE, HIDDEN_SIZE_0, HIDDEN_SIZE_1, OUTPUT_SIZE,
                    file_input_size, file_hidden_size_0, file_hidden_size_1, file_output_size
                )
            ));
        }

        // 3. If structure matches, load parameters
        for param_slice_mut in self.get_params_mut() {
            for float_val_ref in param_slice_mut.iter_mut() {
                file.read_exact(&mut f64_buffer)?;
                *float_val_ref = f64::from_le_bytes(f64_buffer);
            }
        }
        
        // 4. Optional: Check for unexpected extra data after all expected data is read
        let mut one_byte_buffer = [0u8; 1];
        match file.read(&mut one_byte_buffer) {
            Ok(0) => Ok(()), // Expected EOF
            Ok(_) => Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "SMLP file contains more data than expected for the current structure.",
            )),
            Err(e) => Err(e), // Other read error
        }
    }

    // --- Other SMLP methods ---
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

    pub fn predict(&mut self, image: [f64; INPUT_SIZE]) -> SVector<f64, OUTPUT_SIZE> {
        self.input_layer = SVector::<f64, INPUT_SIZE>::from_row_slice(&image);

        self.hidden_layer_0 = (self.weights_matrix_01 * self.input_layer) + self.hidden_bias_0;
        SMLP::apply_relu(&mut self.hidden_layer_0);

        self.hidden_layer_1 = (self.weights_matrix_12 * self.hidden_layer_0) + self.hidden_bias_1;
        SMLP::apply_relu(&mut self.hidden_layer_1);

        self.output_layer = (self.weights_matrix_23 * self.hidden_layer_1) + self.output_bias;
        SMLP::softmax(&self.output_layer)
    }

    pub fn train_cross_entropy_batch(
        &mut self,
        images: &[[f64; INPUT_SIZE]],
        target_labels: &[u8],
        learning_rate: f64,
    ) -> Vec<SVector<f64, OUTPUT_SIZE>> {
        let batch_size = images.len();
        assert_eq!(batch_size, target_labels.len());

        let mut grad_w23_accum = nalgebra::SMatrix::<f64, OUTPUT_SIZE, HIDDEN_SIZE_1>::zeros();
        let mut grad_b_output_accum = nalgebra::SVector::<f64, OUTPUT_SIZE>::zeros();
        let mut grad_w12_accum = nalgebra::SMatrix::<f64, HIDDEN_SIZE_1, HIDDEN_SIZE_0>::zeros();
        let mut grad_b_hidden_1_accum = nalgebra::SVector::<f64, HIDDEN_SIZE_1>::zeros();
        let mut grad_w01_accum = nalgebra::SMatrix::<f64, HIDDEN_SIZE_0, INPUT_SIZE>::zeros();
        let mut grad_b_hidden_0_accum = nalgebra::SVector::<f64, HIDDEN_SIZE_0>::zeros();
        let mut predictions = Vec::with_capacity(batch_size);

        for (image, &label) in images.iter().zip(target_labels.iter()) {
            let prediction = self.predict(*image);
            predictions.push(prediction.clone());

            let mut target = SVector::<f64, OUTPUT_SIZE>::from_element(0.0);
            target[label as usize] = 1.0;

            let delta_output = &prediction - &target;

            let grad_w23 = &delta_output * self.hidden_layer_1.transpose();
            let grad_b_output = delta_output.clone();

            let mut delta_hidden_1 = self.weights_matrix_23.transpose() * &delta_output;
            for i in 0..HIDDEN_SIZE_1 {
                if self.hidden_layer_1[i] <= 0.0 { delta_hidden_1[i] = 0.0; }
            }

            let grad_w12 = &delta_hidden_1 * self.hidden_layer_0.transpose();
            let grad_b_hidden_1 = delta_hidden_1.clone();

            let mut delta_hidden_0 = self.weights_matrix_12.transpose() * &delta_hidden_1;
            for i in 0..HIDDEN_SIZE_0 {
                if self.hidden_layer_0[i] <= 0.0 { delta_hidden_0[i] = 0.0; }
            }

            let grad_w01 = &delta_hidden_0 * self.input_layer.transpose();
            let grad_b_hidden_0 = delta_hidden_0;

            grad_w23_accum += grad_w23;
            grad_b_output_accum += grad_b_output;
            grad_w12_accum += grad_w12;
            grad_b_hidden_1_accum += grad_b_hidden_1;
            grad_w01_accum += grad_w01;
            grad_b_hidden_0_accum += grad_b_hidden_0;
        }

        let batch_size_f64 = batch_size as f64;
        self.weights_matrix_23 -= learning_rate * grad_w23_accum / batch_size_f64;
        self.output_bias -= learning_rate * grad_b_output_accum / batch_size_f64;
        self.weights_matrix_12 -= learning_rate * grad_w12_accum / batch_size_f64;
        self.hidden_bias_1 -= learning_rate * grad_b_hidden_1_accum / batch_size_f64;
        self.weights_matrix_01 -= learning_rate * grad_w01_accum / batch_size_f64;
        self.hidden_bias_0 -= learning_rate * grad_b_hidden_0_accum / batch_size_f64;

        predictions
    }
}