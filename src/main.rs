mod mlp;
mod constants;
mod utils;

use utils::io::MNISTReader;
use std::path::Path;

use crate::mlp::MLP;
use crate::utils::{io, parsers};

fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let train_mnist = MNISTReader::new(
        Path::new("./data/train-images.idx3-ubyte"),
        Path::new("./data/train-labels.idx1-ubyte")
    )?;

    let mut mlp = MLP::new();
    mlp.show_weights();
    mlp.randomize_weights();
    mlp.show_weights();

    for result in train_mnist {
        let (image, label) = result?;
        io::render_mnist_image(&image, 'p');
        println!("LABEL: {}", label);
        println!();        
        //let normalized_image: [f64; 784] = parsers::normalize_image(image);
        //let prediction = mlp.predict(normalized_image);
        //println!("{:?}", prediction);
        //io::render_mlp_output(&prediction);
    }

    Ok(())
}