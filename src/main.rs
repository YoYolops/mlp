mod mlp;
mod constants;
mod utils;

use utils::io::MNISTReader;
use std::path::Path;

use crate::utils::parsers;
use crate::mlp::MLP;

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.01;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mlp = MLP::new();
    mlp.randomize_weights();

    for epoch in 1..=EPOCHS {
        let train_mnist = MNISTReader::new(
            Path::new("./data/train-images.idx3-ubyte"),
            Path::new("./data/train-labels.idx1-ubyte"),
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