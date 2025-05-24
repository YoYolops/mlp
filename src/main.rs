mod dynamic_mlp;
mod static_mlp;
mod constants;
mod utils;

use utils::io::MNISTReader;
use std::path::Path;

use crate::utils::parsers;
use crate::dynamic_mlp::DMLP;
use crate::static_mlp::SMLP;

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.01;

const TRAINING_IMAGES_PATH: &str = "./training_data/train-images.idx3-ubyte";
const TRAINING_LABELS_PATH: &str = "./training_data/train-labels.idx1-ubyte";

fn train_dynamic_mlp() -> Result<(), Box<dyn std::error::Error>> {
    let layer_sizes = vec![784, 16, 16, 10];
    // Weights randomization is done in NEW for the dianmyc MLP
    let mut dmlp = DMLP::new(layer_sizes);
    dmlp.show_layers();

    for epoch in 1..=EPOCHS {
        let train_mnist = MNISTReader::new(
            Path::new(TRAINING_IMAGES_PATH),
            Path::new(TRAINING_LABELS_PATH),
        )?;

        let mut images_batch = Vec::with_capacity(BATCH_SIZE);
        let mut labels_batch = Vec::with_capacity(BATCH_SIZE);
        let mut correct = 0;
        let mut total = 0;

        for result in train_mnist {
            let (image, label) = result?;
            let normalized = parsers::normalize_image(image);
            images_batch.push(normalized.to_vec()); // Conversão necessária
            labels_batch.push(label);

            if images_batch.len() == BATCH_SIZE {
                let predictions = dmlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);

                for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                    if pred.argmax().0 == actual_label as usize {
                        correct += 1;
                    }
                    total += 1;
                }

                images_batch.clear();
                labels_batch.clear();
            }
        }

        // Handle leftovers
        if !images_batch.is_empty() {
            let predictions = dmlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);
            for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                if pred.argmax().0 == actual_label as usize {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = (correct as f64) / (total as f64) * 100.0;
        println!("Epoch {epoch} Accuracy: {:.2}%", accuracy);
    }

    Ok(())
}

fn train_static_mlp() -> Result<(), Box<dyn std::error::Error>> {
    let mut mlp = SMLP::new();
    mlp.randomize_weights();

    for epoch in 1..=EPOCHS {
        let train_mnist = MNISTReader::new(
            Path::new(TRAINING_IMAGES_PATH),
            Path::new(TRAINING_LABELS_PATH),
        )?;

        let mut images_batch = Vec::with_capacity(BATCH_SIZE);
        let mut labels_batch = Vec::with_capacity(BATCH_SIZE);
        let mut correct = 0;
        let mut total = 0;

        for result in train_mnist {
            let (image, label) = result?;
            let normalized = parsers::normalize_image(image);
            images_batch.push(normalized);
            labels_batch.push(label);

            if images_batch.len() == BATCH_SIZE {
                let predictions = mlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);

                for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                    if pred.argmax().0 == actual_label as usize {
                        correct += 1;
                    }
                    total += 1;
                }

                images_batch.clear();
                labels_batch.clear();
            }
        }

        // Handle leftovers
        if !images_batch.is_empty() {
            let predictions = mlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);
            for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                if pred.argmax().0 == actual_label as usize {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = (correct as f64) / (total as f64) * 100.0;
        println!("Epoch {epoch} Accuracy: {:.2}%", accuracy);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HEY THERE");
    train_static_mlp()?;
    //train_dynamic_mlp()?;
    Ok(())
}