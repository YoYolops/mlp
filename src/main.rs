mod mlp;
mod constants;
mod utils;

use utils::io::MNISTReader;
use std::path::Path;

use crate::mlp::MLP;
use crate::utils::{io, parsers};

fn main() -> Result<(), Box<dyn std::error::Error>>  {
    const TRAINING_EPOCHS: u32 = 10;
    let mut mlp = MLP::new();

    mlp.randomize_weights();
    mlp.show_weights();

    for epoch in 0..TRAINING_EPOCHS {
        let train_mnist = MNISTReader::new(
            Path::new("./data/train-images.idx3-ubyte"),
            Path::new("./data/train-labels.idx1-ubyte")
        )?;
        let mut correct_predictions: i32 = 0;
        let mut total_predictions: i32 = 0;

        for result in train_mnist {
            let (image, label) = result?;
            //io::render_mnist_image(&image, 'p');
            let normalized_image: [f64; 784] = parsers::normalize_image(image);
            let prediction = mlp.train_cross_entropy(normalized_image, label, 0.1);
            if prediction.argmax().0 == label as usize {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }
        println!("Epoch {epoch} Accuracy: {:.2}%", 100.0 * correct_predictions as f64 / total_predictions as f64);
    }

    Ok(())
}