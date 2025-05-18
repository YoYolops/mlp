pub fn normalize_image(image: [u8; 784]) -> [f64; 784] {
    let mut normalized_image: [f64; 784] = [0.0; 784];

    for (i, &pixel_byte) in image.iter().enumerate() {
        normalized_image[i] = pixel_byte as f64 / 255.0;
    }
    
    normalized_image
}