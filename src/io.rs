use std::fs::File;
use std::io::{self, BufReader, Read};

pub fn read_image() -> io::Result<()> {
    let file = File::open("./data/train-images.idx3-ubyte")?;
    let mut reader = BufReader::new(file);

    let magic = read_u32_big_endian(&mut reader)?;
    if magic != 2051 {
        panic!("Invalid magic number: {}", magic);
    }

    let num_images = read_u32_big_endian(&mut reader)?;
    let num_rows = read_u32_big_endian(&mut reader)?;
    let num_cols = read_u32_big_endian(&mut reader)?;

    println!("Images: {}, Size: {}x{}", num_images, num_rows, num_cols);

    let image_size = (num_rows * num_cols) as usize;

    for i in 0..num_images {
        let mut image = vec![0u8; image_size];
        println!("Reading image: {}", i+1);

        reader.read_exact(&mut image)?;
        render_image(&image, 'p');
    }

    Ok(())
}

// Helper to read a u32 in big-endian format
fn read_u32_big_endian(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn render_image(image: &[u8], mode: char) {
    const WIDTH: usize = 28;
    const HEIGHT: usize = 28;
    const GREYSCALE: [char; 5] = ['█', '▓', '▒', '░', ' ']; // dark to light
    const NEGATIVE_GREYSCALE: [char; 5] = [' ', '░', '▒', '▓', '█'];

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = image[y * WIDTH + x];
            let idx = ((pixel as usize * (GREYSCALE.len() - 1)) / 255) as usize;
            let block = if mode == 'n' {
                NEGATIVE_GREYSCALE[NEGATIVE_GREYSCALE.len() - 1 - idx]
            } else {
                GREYSCALE[GREYSCALE.len() - 1 - idx]
            };
            
            print!("{}", block);
        }
        println!();
    }
}