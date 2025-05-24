mod dynamic_mlp;
mod static_mlp;
mod constants;
mod utils;

use utils::io::{
    MNISTReader,
    InputHandler,
    render_mnist_image,
    render_mlp_output
};
use std::path::Path;

use crate::utils::parsers;
use crate::dynamic_mlp::DMLP;
use crate::static_mlp::SMLP;
use crate::constants::{
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRAINING_IMAGES_PATH,
    TRAINING_LABELS_PATH,
    SMLP_WEIGHTS_PATH,
    INPUT_FOLDER_PATH
};

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
    let mut smlp = SMLP::new();
    smlp.randomize_weights();

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
                let predictions = smlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);

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
            let predictions = smlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);
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
    println!("Saving weights in {}", SMLP_WEIGHTS_PATH);
    smlp.save_weights(SMLP_WEIGHTS_PATH)?;

    Ok(())
}

/* fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut smlp: SMLP = SMLP::new();
    let input_handler = InputHandler::new(INPUT_FOLDER_PATH)?;

    smlp.load_weights(SMLP_WEIGHTS_PATH)?;
    smlp.show_weights();

    for image_result in input_handler {
        match image_result {
            Ok(image) => {
                render_mnist_image(&image, 'p');
                let normalized_image = parsers::normalize_image(image);
                let prediction = smlp.predict(normalized_image);
                render_mlp_output(&prediction);
            },
            Err(e) => eprintln!("Erro ao carregar imagem: {}", e),
        }
    }

    Ok(())
} */

fn main() -> Result<(), Box<dyn std::error::Error>> {
    train_static_mlp()?;
    Ok(())
}