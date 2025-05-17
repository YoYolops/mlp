use std::fs::File;
use std::io::{BufReader, Read};
use byteorder::{ReadBytesExt, BigEndian};

pub fn read_image() -> std::io::Result<()> {
    let file: File = File::open("./data/train-images.idx3-ubyte")?;
    let mut reader: BufReader<File> = BufReader::new(file);

    let magic: u32 = reader.read_u32::<BigEndian>()?;
    if magic != 2051 {
        panic!("Invalid magic number: {}", magic);
    }

    let num_images: u32 = reader.read_u32::<BigEndian>()?;
    let num_rows: u32 = reader.read_u32::<BigEndian>()?;
    let num_cols: u32 = reader.read_u32::<BigEndian>()?;

    println!("Images: {}, Size: {}x{}", num_images, num_rows, num_cols);

    let image_size: usize = (num_rows * num_cols) as usize;
    
    for i in 0..num_images {
        let mut image: Vec<u8> = vec![0u8; image_size];
        println!("{}", i);
        
            // Read and render the first image
            reader.read_exact(&mut image)?;
            render_image(&image);
    }

    Ok(())
}

fn render_image(image: &[u8]) {
    let width = 28;
    let height = 28;
    let blocks = ['█', '▓', '▒', '░', ' ']; // dark to light

    for y in 0..height {
        for x in 0..width {
            let pixel = image[y * width + x];
            // Map 0..=255 to 0..=blocks.len()-1 (inverted: 0=darkest)
            let idx = ((pixel as usize * (blocks.len() - 1)) / 255) as usize;
            // Invert index so that 0 (black) => blocks[0] (█)
            let block = blocks[blocks.len() - 1 - idx];
            print!("{}", block);
        }
        println!();
    }
}