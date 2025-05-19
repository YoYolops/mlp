mod mlp;
mod constants;
mod utils;

use crate::mlp::MLP;
use crate::utils::{io, parsers};



fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let mut mlp = MLP::new();
    mlp.show_weights();
    mlp.randomize_weights();
    mlp.show_weights();
    let image: [u8; 784] = io::read_image()?;
    let normalized_image: [f64; 784] = parsers::normalize_image(image);
    let prediction = mlp.predict(normalized_image);
    println!("{:?}", prediction);
    io::render_output(&prediction);
    Ok(())
}
