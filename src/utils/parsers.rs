use crate::constants::INPUT_SIZE;

pub fn normalize_image(image: [u8; INPUT_SIZE]) -> [f64; INPUT_SIZE] {
    let mut normalized_image: [f64; INPUT_SIZE] = [0.0; INPUT_SIZE];

    for (i, &pixel_byte) in image.iter().enumerate() {
        normalized_image[i] = pixel_byte as f64 / 255.0;
    }

    normalized_image
}