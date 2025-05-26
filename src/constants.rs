// Static mlp (SMLP) structure
pub const INPUT_SIZE: usize = 784;
pub const HIDDEN_SIZE_0: usize = 112;
pub const HIDDEN_SIZE_1: usize = 16;
pub const OUTPUT_SIZE: usize = 10;
pub const LABEL_SIZE: usize = 1;

// Training params
pub const BATCH_SIZE: usize = 32;
pub const EPOCHS: usize = 10;
pub const LEARNING_RATE: f64 = 0.01;

// Relevant Paths
pub const TRAINING_IMAGES_PATH: &str = "./training_data/train-images.idx3-ubyte";
pub const TRAINING_LABELS_PATH: &str = "./training_data/train-labels.idx1-ubyte";

pub const SMLP_WEIGHTS_PATH: &str = "./mlp_params/smlp/smlp_weights.bin";
pub const DMLP_WEIGHTS_PATH: &str = "./mlp_params/dmlp/dmlp_weights.bin";
pub const INPUT_FOLDER_PATH: &str = "./input_data";