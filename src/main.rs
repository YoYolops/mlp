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
use std::thread;
use rand::seq::SliceRandom;

use crate::utils::parsers;
use crate::dynamic_mlp::DMLP;
use crate::static_mlp::SMLP;
use crate::constants::{
    INPUT_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRAINING_IMAGES_PATH,
    TRAINING_LABELS_PATH,
    SMLP_WEIGHTS_PATH,
    DMLP_WEIGHTS_PATH,
    INPUT_FOLDER_PATH,
};

fn train_dynamic_mlp() -> Result<(), Box<dyn std::error::Error>> {
    let layer_sizes = vec![784, 16, 16, 10];
    // Weights randomization is done in NEW for the dynamic MLP
    let mut dmlp = DMLP::new(layer_sizes.clone()); // Pass layer_sizes by value or clone if needed later
    println!("Dynamic MLP Layers:"); // Changed from dmlp.show_layers() to avoid assuming it prints
    print!("[");
    for (i, &size) in layer_sizes.iter().enumerate() {
        print!("{}", size);
        if i < layer_sizes.len() - 1 {
            print!(", ");
        }
    }
    println!("]");


    for epoch in 1..=EPOCHS {
        // 1. Load and Prepare Full Dataset for the current epoch
        let train_mnist_reader = MNISTReader::new(
            Path::new(TRAINING_IMAGES_PATH),
            Path::new(TRAINING_LABELS_PATH),
        )?;

        let mut dataset: Vec<(Vec<f64>, u8)> = train_mnist_reader
            .map(|result| {
                let (image_raw, label) = result?; // Assuming MNISTReader yields Result<(RawImage, u8)>
                let normalized_image_array = parsers::normalize_image(image_raw); // Assuming this returns [f64; 784]
                Ok::<_, Box<dyn std::error::Error>>((
                    normalized_image_array.to_vec(), // Convert [f64; 784] to Vec<f64>
                    label,
                ))
            })
            .collect::<Result<Vec<(Vec<f64>, u8)>, _>>()?;

        // 2. Shuffle Dataset
        let mut rng = rand::rng();
        dataset.shuffle(&mut rng);

        let mut epoch_correct = 0;
        let mut epoch_total = 0;

        // 3. Process in Batches
        for chunk in dataset.chunks(BATCH_SIZE) {
            // 4. Adapt Batch Creation
            // .cloned() is used because chunk elements are &(Vec<f64>, u8)
            // and we need owned Vec<f64> and u8 for the new vectors.
            let (images_batch, labels_batch): (Vec<Vec<f64>>, Vec<u8>) = 
                chunk.iter().cloned().unzip();

            if images_batch.is_empty() { // Should not happen if dataset is not empty
                continue;
            }

            // Train on the current batch
            let predictions = dmlp.train_cross_entropy_batch(
                &images_batch, 
                &labels_batch, 
                LEARNING_RATE
            );

            // Calculate accuracy for this batch (optional, or accumulate for epoch accuracy)
            for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                if pred.argmax().0 == actual_label as usize {
                    epoch_correct += 1;
                }
                epoch_total += 1;
            }
        } // End of batch processing loop

        let accuracy = if epoch_total > 0 {
            (epoch_correct as f64) / (epoch_total as f64) * 100.0
        } else {
            0.0 // Avoid division by zero if dataset was empty
        };
        println!("Epoch {epoch:02} Accuracy: {:.2}% (Correct: {}, Total: {})", accuracy, epoch_correct, epoch_total);
    } // End of epoch loop

    // Optionally save weights for DMLP if you have a path defined
    println!("Saving dynamic MLP weights...");
    dmlp.save_weights(DMLP_WEIGHTS_PATH)?; // Assuming DMLP_WEIGHTS_PATH is defined

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
        let mut dataset: Vec<([f64; INPUT_SIZE], u8)> = train_mnist
            .map(|res| {
                let (img, label) = res?;
                Ok::<_, Box<dyn std::error::Error>>((
                    parsers::normalize_image(img),
                    label,
                ))
            })
            .collect::<Result<_, _>>()?;

        // Shuffle dataset
        dataset.shuffle(&mut rand::rng());

        let mut correct = 0;
        let mut total = 0;
        for chunk in dataset.chunks(BATCH_SIZE) {
            let (images_batch, labels_batch): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
            let predictions = smlp.train_cross_entropy_batch(&images_batch, &labels_batch, LEARNING_RATE);

            for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                if pred.argmax().0 == actual_label as usize {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = (correct as f64) / (total as f64) * 100.0;
        println!("Epoch {epoch:02} Accuracy: {:.2}%", accuracy);
    }
    println!("Saving weights in {}", SMLP_WEIGHTS_PATH);
    smlp.save_weights(SMLP_WEIGHTS_PATH)?;

    Ok(())
}

/* fn run() -> Result<(), Box<dyn std::error::Error>> {
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    train_static_mlp()?;
    Ok(())
}

fn main() {
    thread::Builder::new()
        .stack_size(16 * 1024 * 1024 * 2) // 32 MB
        .spawn(|| {
            if let Err(e) = run() {
                eprintln!("Error: {e}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}