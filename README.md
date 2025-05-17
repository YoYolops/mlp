# Multi Layer Perceptron - MLP
Today is may 16th, 2025. I'll **TRY** to build a multilayer perceptron from the ground up with rust. This repository will only go public if i succeed :)

### THE PLAN 
I will use [the MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) to train my perceptron to be able to recognize handwritten digits.

What can go wrong? The MLP general architecture is fairly simple and this problem in specific was already tackled many times, so there will be lots of reference material available.

This entire project will be mainly based on [three blue one brown's video series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).

### Step one: Get the data

To my surprise, the MNIST dataset comes in an alien format. It appears to be provided as raw binary big-endian integers (?). I was expecting a folder with tons of 28x28 jpegs and this starts to bother me a little bit. How am i supposed to read this?

After being in shock for a while, a did some research and it turns out the binary format of the MINIST dataset will make my job A LOT easier.

¯\\_(ツ)_/¯

### Reading The Data
After reading [a stack overflow post](https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format) and asking some questions to gepetto (chat gpt's pet name), i kinda have figured out how to read the images.

These are the first bytes. They're headers giving general information about the following dataset.

Offset*| Meaning
-------|----------------------------
0–3    | Magic number      (2051)
4–7    | Number of images  (60000)
8–11   | Number of rows    (28)
12–15  | Number of columns (28)

*the offset is in bytes

After that, we will have all the 28x28 bitmap images, where each pixel is a single byte integer (0-255), representing a greyscale. That's why i said the binary format of the dataset with its bitmap greyscale nature would come in hand. jpeg images would need to be pre processed to turn each pixel into an integer, but now we can skip this part. It's just like if the dataset was specially designed to this algorithm.

Since Rust already provides functionalities to read binary data, i just had to write a simple wrapper function that receives a reader and starts appending 32bits integers to it:

```rust
fn read_u32_big_endian(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}
```

After opening the file and moving the buffer cursor away from the headers and start reading the images, its time to render them. For that, we get each pixel integer value and render one of five ascii characters representing different greyscales:

```rust
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
```

Now we are able to see them on terminal:

