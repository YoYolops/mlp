mod mlp;
mod constants;
mod utils;

use crate::mlp::MLP;
use crate::utils::{io, parsers};


fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let mut mlp = MLP::new();
    mlp.load_weights();
    mlp.show_weights();
    mlp.randomize_weights();
    mlp.show_weights();
    let image = io::read_image()?;
    let normalized_image = parsers::normalize_image(image);
    mlp.predict(normalized_image);
    Ok(())
}
