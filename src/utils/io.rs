use std::fs::{File, read_dir};
use std::io::{self, BufReader, Read};
use std::vec::IntoIter;
use nalgebra::SVector;
use std::path::{Path, PathBuf};

use image::ImageReader ;
use std::process::Command;
use image::{GrayImage, GenericImageView};
use tempfile::NamedTempFile;

use crate::constants::{INPUT_SIZE, OUTPUT_SIZE, LABEL_SIZE};

pub struct InputHandler {
    image_paths: IntoIter<PathBuf>,
}

impl InputHandler {
    pub fn new<P: AsRef<Path>>(folder: P) -> io::Result<Self> {
        let entries = read_dir(&folder)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().map_or(false, |ext| ext == "png")
            })
            .collect::<Vec<_>>();

        Ok(
            InputHandler {
                image_paths: entries.into_iter(),
            }
        )
    }
    
}

impl Iterator for InputHandler {
    type Item = Result<[u8; INPUT_SIZE], Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.image_paths.next().map(|path| png_to_mnist(path))
    }
}

pub struct MNISTReader {
    image_reader: BufReader<File>,
    label_reader: BufReader<File>,
    index: usize,
    total: usize,
}

impl MNISTReader {
    pub fn new(image_path: &Path, label_path: &Path) -> io::Result<Self> {
        let mut image_file = BufReader::new(File::open(image_path)?);
        let mut label_file = BufReader::new(File::open(label_path)?);

        // Skiping the headers
        let mut image_header = [0u8; 16];
        let mut label_header = [0u8; 8];
        image_file.read_exact(&mut image_header)?;
        label_file.read_exact(&mut label_header)?;

        let total = u32::from_be_bytes([image_header[4], image_header[5], image_header[6], image_header[7]]) as usize;

        Ok(
            MNISTReader {
                image_reader: image_file,
                label_reader: label_file,
                index: 0,
                total,
            }
        )
    }
}

impl Iterator for MNISTReader {
    type Item = io::Result<([u8; INPUT_SIZE], u8)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }

        let mut image = [0u8; INPUT_SIZE];
        let mut label = [0u8; LABEL_SIZE];

        if let Err(e) = self.image_reader.read_exact(&mut image) {
            return Some(Err(e));
        }

        if let Err(e) = self.label_reader.read_exact(&mut label) {
            return Some(Err(e));
        }

        self.index += 1;
        Some(Ok((image, label[0])))
    }
}

pub fn render_mnist_image(image: &[u8], mode: char) {
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

pub fn png_to_mnist<P: AsRef<Path>>(path: P) -> Result<[u8; INPUT_SIZE], Box<dyn std::error::Error>> {
    let input_path = path.as_ref();

    // Create a temporary file path with a .png extension
    let temp_output_file_base = NamedTempFile::new()?;
    let mut temp_output_path: PathBuf = temp_output_file_base.path().to_path_buf();
    temp_output_path.set_extension("png"); 
    
    let _temp_file_handle = temp_output_file_base;

    // Builds the FFmpeg command
    let status = Command::new("ffmpeg")
        .arg("-i")
        .arg(input_path)
        .arg("-vf")
        .arg("scale=28:28,format=gray") // Resize to 28x28 and convert to grayscale
        .arg("-y") // Overwrite output file without asking
        .arg(&temp_output_path) // Use the path with the extension
        .status()?;

    if !status.success() {
        return Err(format!("FFmpeg command failed with exit code: {:?}", status.code()).into());
    }

    // Open the processed image from the temporary file using the image crate
    let img = ImageReader::open(&temp_output_path)?.decode()?.to_luma8();

    let mut data = [0u8; INPUT_SIZE];
    for (i, pixel) in img.pixels().enumerate() {
        data[i] = pixel[0];
    }

    // The temporary file will be automatically deleted when _temp_file_handle goes out of scope.
    Ok(data)
}