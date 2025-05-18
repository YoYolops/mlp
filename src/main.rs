mod mlp;

use crate::mlp::MLP;

fn main() {
    let mut mlp = MLP::new();
    mlp.load_weights();
    mlp.show_weights();
    mlp.randomize_weights();
    mlp.show_weights();
}
