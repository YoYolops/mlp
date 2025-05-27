mod dynamic_mlp;
mod static_mlp;
mod constants;
mod utils;

use utils::io::{
    MNISTReader,
    InputHandler,
    render_mnist_image,
};
use std::path::Path;
use std::{thread};
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
    DMLP_LAYERS
};

fn train_dynamic_mlp() -> Result<(), Box<dyn std::error::Error>> {
    let layer_sizes = Vec::from(DMLP_LAYERS);
    // Weights randomization is done in NEW for the dynamic MLP
    let mut dmlp = DMLP::new(layer_sizes);
    println!("Dynamic MLP Layers:");
    dmlp.show_layers();

    for epoch in 1..=EPOCHS {
        let train_mnist_reader: MNISTReader = MNISTReader::new(
            Path::new(TRAINING_IMAGES_PATH),
            Path::new(TRAINING_LABELS_PATH),
        )?;

        let mut dataset: Vec<(Vec<f64>, u8)> = train_mnist_reader
            .map(|result| {
                let (image_raw, label) = result?;
                let normalized_image_array = parsers::normalize_image(image_raw);
                Ok::<_, Box<dyn std::error::Error>>((
                    normalized_image_array.to_vec(), // Convert [f64; 784] to Vec<f64>
                    label,
                ))
            })
            .collect::<Result<Vec<(Vec<f64>, u8)>, _>>()?;

        // Shuffle Dataset
        let mut rng = rand::rng();
        dataset.shuffle(&mut rng);

        let mut correct = 0;
        let mut total = 0;

        // Batches
        for chunk in dataset.chunks(BATCH_SIZE) {
            // .cloned() is used because chunk elements are &(Vec<f64>, u8)
            // and we need owned Vec<f64> and u8 for the new vectors.
            let (images_batch, labels_batch): (Vec<Vec<f64>>, Vec<u8>) = 
                chunk.iter().cloned().unzip();

            if images_batch.is_empty() {
                continue;
            }

            // Train on the current batch
            let predictions = dmlp.train_cross_entropy_batch(
                &images_batch, 
                &labels_batch, 
                LEARNING_RATE
            );

            // Calculate accuracy for this batch
            for (pred, &actual_label) in predictions.iter().zip(&labels_batch) {
                if pred.argmax().0 == actual_label as usize {
                    correct += 1;
                }
                total += 1;
            }
        } // End of batch processing loop

        let accuracy = if total > 0 {
            (correct as f64) / (total as f64) * 100.0
        } else {
            0.0 // Avoid division by zero if dataset was empty
        };
        println!("Epoch {epoch:02} Accuracy: {:.2}%", accuracy);
    } // End of epoch loop

    println!("Saving DMLP weights in {}", DMLP_WEIGHTS_PATH);
    dmlp.save_weights(DMLP_WEIGHTS_PATH)?;

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
    println!("Saving SMLP weights in {}", SMLP_WEIGHTS_PATH);
    smlp.save_weights(SMLP_WEIGHTS_PATH)?;

    Ok(())
}

fn test_smlp_againt_input() -> Result<(), Box<dyn std::error::Error>> {
    let mut smlp: SMLP = SMLP::new();
    smlp.load_weights(SMLP_WEIGHTS_PATH)?;
    smlp.show_weights();
    
    let input_handler = InputHandler::new(INPUT_FOLDER_PATH)?;


    for image_result in input_handler {
        match image_result {
            Ok(image) => {
                render_mnist_image(&image, 'p');
                let normalized_image = parsers::normalize_image(image);
                let prediction = smlp.predict(normalized_image);
                smlp.render_output(&prediction);
            },
            Err(e) => eprintln!("Erro ao carregar imagem: {}", e),
        }
    }

    Ok(())
}

fn test_dmlp_againt_input() -> Result<(), Box<dyn std::error::Error>> {
    let mut dmlp = DMLP::new(Vec::from(DMLP_LAYERS));

    println!("Loading DMLP weights from: {}", DMLP_WEIGHTS_PATH);
    dmlp.load_weights(DMLP_WEIGHTS_PATH)?;

    println!("DMLP Layers Structure:");
    dmlp.show_layers();

    println!("\nInitializing Input Handler for folder: {}", INPUT_FOLDER_PATH);
    let input_handler = InputHandler::new(Path::new(INPUT_FOLDER_PATH))?;

    println!("\nProcessing images from input folder...");
    for image_result in input_handler {
        match image_result {
            Ok(image_data) => { // Assuming image_data is the raw image type
                render_mnist_image(&image_data, 'p'); // Render the input image

                // parsers::normalize_image should return [f64; INPUT_SIZE]
                let normalized_image_array = parsers::normalize_image(image_data);

                // DMLP::predict takes [f64; INPUT_SIZE] by value
                let prediction_dvector = dmlp.predict(normalized_image_array);

                // Call the DMLP's own render method
                dmlp.render_output(&prediction_dvector);
            }
            Err(e) => {
                eprintln!("Error loading or processing image: {}", e);
            }
        }
    }

    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    test_dmlp_againt_input()?;
    Ok(())
}

fn main() {
    thread::Builder::new()
        .stack_size(16 * 1024 * 1024 * 2) // 32 MB of stack allocated
        .spawn(|| {
            if let Err(e) = run() {
                eprintln!("Error: {e}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}