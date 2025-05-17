mod io;
mod mlp;

use crate::mlp::MLP;

fn main() -> std::io::Result<()> {
    let mlp = MLP::new();
    mlp.load_weights();
    io::read_image()
}
