use rand::rng;
use rand_distr::{Distribution, Normal};
use nalgebra::{DMatrix, DVector};

pub struct DMLP {
    layers: Vec<usize>,
    activations: Vec<DVector<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DVector<f64>>,
}

impl DMLP {
    pub fn new(layers: Vec<usize>) -> Self {
        let mut rng = rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();

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

        for &layer_size in &layers {
            activations.push(DVector::from_element(layer_size, 0.0));
        }

        DMLP { layers, weights, biases, activations }
    }

    pub fn predict(&mut self, input: &[f64]) -> DVector<f64> {
        self.activations[0] = DVector::from_row_slice(input);

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
        let batch_size = images.len();
        let mut predictions = Vec::with_capacity(batch_size);

        let mut grad_w: Vec<DMatrix<f64>> = self.weights.iter()
            .map(|w| DMatrix::zeros(w.nrows(), w.ncols())).collect();

        let mut grad_b: Vec<DVector<f64>> = self.biases.iter()
            .map(|b| DVector::zeros(b.len())).collect();

        for (x, &label) in images.iter().zip(labels.iter()) {
            let y_hat = self.predict(x);
            predictions.push(y_hat.clone());

            let mut target = DVector::from_element(y_hat.len(), 0.0);
            target[label as usize] = 1.0;

            let mut delta = &y_hat - &target;

            for l in (0..self.weights.len()).rev() {
                let a_prev = &self.activations[l];
                grad_w[l] += &delta * a_prev.transpose();
                grad_b[l] += &delta;

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

        let rate = learning_rate / batch_size as f64;
        for i in 0..self.weights.len() {
            self.weights[i] -= &grad_w[i] * rate;
            self.biases[i] -= &grad_b[i] * rate;
        }

        predictions
    }

    pub fn show_weights(&self) {
        for (i, w) in self.weights.iter().enumerate() {
            println!("Weights between layer {} and {}:", i, i + 1);
            println!("{}", w);
        }
    }

    fn relu(x: &DVector<f64>) -> DVector<f64> {
        x.map(|v| v.max(0.0))
    }

    fn softmax(x: &DVector<f64>) -> DVector<f64> {
        let max = x.max();
        let exps = x.map(|v| (v - max).exp());
        let sum: f64 = exps.iter().sum();
        exps / sum
    }

    pub fn show_layers(&self) {
        println!("Layers: {:#?}", self.layers);
    }

    pub fn load_weights(&self) {
        println!("Not implemented yet.");
    }

    pub fn save_weights(&self) {
        println!("Not implemented yet.");
    }
}
