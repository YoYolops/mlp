# Multilayer Perceptron - MLP
Today is may 16th, 2025. I'll **TRY** to build a multilayer perceptron from the ground up with rust. This repository will only go public if i succeed :)

### THE PLAN 
I will use [the MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) to train my perceptron to be able to recognize handwritten digits.

What can go wrong? The MLP general architecture is fairly simple and this problem in specific was already tackled many times, so there will be lots of reference material available.

This entire project will be mainly based on [three blue one brown's video series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). So every time i refer to our **"research source"**, i'll be talking about this series. 

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

Now we are able to see them on terminal (the green background is my terminal's):

![Captura de tela de 2025-05-17 02-00-20](https://github.com/user-attachments/assets/d696e2ee-85aa-49e1-9b24-a8368f278885)

And that is clearly a two, ladies and gentleman.

### Discussing the MLP architecture

As the name suggests, the Multilayer Perceptron has multiple layers (🤯). Ours will have 4 layers, which must be enough to the task (we can experiment multiple variants later on). The first layer will be composed by 784 neurons (the amount of an image's pixels), followed by two 16-neuron layers, and a final 10 neuron layer, corresponding to each possible label (0 to 9).
We will not focus on models scalability, although it would be nice to have a flexibe MLP, with a customizable number and size of layers so on so forth. Our first goal is simply to build the perceptron and correctly classify the dataset images (with good accuracy i mean), so here it goes our base struct:

```rust
pub struct MLP {
    input_layer: [f64; 784],
    hidden_layer_0: [f64; 16],
    hidden_layer_1: [f64; 16],
    output_layer: [f64; 10],
    weights_matrix: [[f64; 784]; 16]
}

impl MLP {

    pub fn new() -> Self {
        MLP {
            input_layer: [0.0; 784],
            hidden_layer_0: [0.0; 16],
            hidden_layer_1: [0.0; 16],
            output_layer: [0.0, 10],
            weights_matrix: [[0.0; 784]; 16]
        }
    }

}
```

I chose f64 numbers to have a nice precision, although having a feeling that this is overkill. I am also concerned about potencial performance issues with those 64bits number all over the neurons, weights and biases, but will see how it goes...

### Building connections
According to my **source**, the connections between neurons are, internally, simply a number that defines how strongly related they are. Since each neuron from a layer connects with all the ones in the next, we are going to have a lot of those f64 numbers flying aroung. (784x16) + (16x16) + (16*10) = 12960 weight values just between the first two layers (with more from neurons themselves) 
As recommended, we are going to represent them in a single matrix: `weights_matrix: [[f64; 784]; 16]`

### Randomizing Weights
The first thing we need to do is randomize the weights matrix. Currently they're all set to zero, which results in dead activation between all conections. A dead brain, so to speak, since all the operations would result in zeros. Unfortunately, rust does not provide a randomizer solution, so we're going to use an external crate, called rand.

We also create a print function, so here it is one of our weights matrix:

```rust
Weights Between Layers 1 & 2:
   0.4   0.6  -0.9  -0.9   0.5  -0.7   0.1  -0.1  -0.8   0.7  -0.1   0.0   0.7  -0.0   0.1  -0.4
  -0.3   0.4  -0.5   0.9   0.1   0.3  -0.6  -0.9   0.8   0.8   0.7   0.7  -0.4  -1.0   0.2   0.6
   0.5  -0.8   0.0   0.6   0.6  -0.9  -0.7  -0.9   0.4   0.2   0.7  -1.0  -0.4  -0.3   0.8   0.5
   0.2   0.8  -0.7  -0.0  -0.5   0.7  -0.2   0.3   0.6  -0.6  -0.9  -0.3  -0.2  -0.1  -0.7  -0.1
  -0.6   0.5  -0.7  -0.6   0.6   0.6   0.4   0.6  -0.1   0.5  -0.2   0.8   0.6  -0.1  -0.4   0.1
   0.3   0.9  -0.2   0.0  -0.0   0.5   0.7  -0.5   0.2  -0.9   0.9  -0.0  -1.0   0.7   0.5  -0.3
  -0.0   0.4  -0.8   0.4  -0.1   0.8  -0.4   0.1   0.6   0.3  -0.0  -0.1  -0.5  -0.0   0.7   0.2
  -0.7   0.6   1.0   0.9   0.8   0.0  -0.6   0.1  -0.9  -0.8   0.6  -0.8  -0.4  -0.7  -0.7  -0.4
   0.7   0.3   0.6   0.7  -0.8  -0.7  -0.8   0.1   0.8  -0.8  -0.9  -0.6   0.4  -0.6  -0.8   0.4
   0.3  -0.8  -0.1  -0.6   1.0  -0.8  -0.2  -0.6   0.3  -0.3  -0.5  -0.9  -0.6   0.5  -0.5  -0.2
   0.6  -0.6   0.1   0.8   0.1  -0.6   0.2  -0.5   0.8  -0.8  -0.9   0.9  -0.4   0.8  -0.0   0.2
   0.8   0.9  -0.1   0.8   0.7   0.2   0.7   0.7  -0.3  -0.7  -0.7   0.1  -0.6   0.1   0.3   0.5
  -0.6   0.1  -0.9  -0.3   0.1   1.0   0.3  -0.8   0.1  -0.0  -0.5   0.5  -1.0   0.2   0.9   0.3
  -0.6   0.5   0.3  -0.3   0.8   0.4   0.2   0.9   0.3   0.5   0.2  -0.4   0.4   0.7  -0.6  -0.8
  -0.7   1.0  -0.8   0.7   0.1   0.2  -0.4   0.9   0.8  -0.4  -0.6  -0.3  -0.8   0.1   0.3  -0.4
  -0.3   0.3   0.5  -0.1  -0.8   0.8  -0.3  -0.1  -1.0  -0.6   0.8   0.0  -1.0  -0.1   0.3   1.0
```

### Perceptron Reborn
As of now, our baby perceptron is, at a structural level, entirely built! We already have all the neurons and connections set up. The next step is to create a couple of things to make it functional, eg. the activation function, feed foward etc. Before that, we need to go back to our `io.rs` module. 

### The Feeding Data Problem (Or making the perceptron see)
The entire purpose of our first perceptron layer is to represent the image it's currently being analyzed (or looked at), hence the layer size being tightly linked to the image's. Here we have a mismatch. Each neuron receives data from a pixel, which is encoded in a 8bit number, ranging from 0-255. But a neuron will only hold values between 0 and 1. We need a way of maping those pixels values to something that can be represented inside our neurons.

Since values between 0 and 1 can be understood as a percentage, and a pixel is always in between the concepts of entirely lit (at its brightest, 100%) and entirely dimmed (at its darkest, 0%), we can obtain the percentage of how much bright each pixel has by simply dividing its value to 255. Now we have a normalizing function:

```rust
pub fn normalize_image(image: [u8; 784]) -> [f64; 784] {
    let mut normalized_image: [f64; 784] = [0.0; 784];

    for (i, &pixel_byte) in image.iter().enumerate() {
        normalized_image[i] = pixel_byte as f64 / 255.0;
    }
    
    normalized_image
}
```

### Early Performance Concern
I'm not sure if this perceptron will be that hard to process, since it is very simple. But even this simple perceptron will require millions, maybe billions of matrices operations. I could implement methods for those operations myself, but i will use a library instead, who implements performance improvements in those kinds of operations, potencially reducing training time. After a couple of minutes i've just adapted a few files to make use of our new library instead of pure arrays to store matrices.

### Our Perceptron's First Toughts
With all of our neurons and connections setup. Functions to read, normalize and feed the dataset to the input layer, and all the matrices operations setup, our perceptron is finally able to see and (try to) classify images, despite being really dump right now. Also, a few normalizing operations are still missing, so we can't quite extract any kind of reason from its outputs (yet).

In addittion to that, we are still missing a lot of training functions and related features to support this process. I'll be handling this very soon.

As discussed before, while we were creating the function to initialize the weights matrices with random numbers, if our perceptron starts its weights as zeros, the output will always be a vector of zeros:

```rust
// Output Neurons
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
```

After randomizing our weights:
```rust
[[2.832064153297856, -7.189384950817572, 29.26444281394776, 38.91743873415687, -15.684375189411712, -11.421555730798271, -3.4437869072569565, -20.747108787807665, 5.599649623093166, -9.700232613382788]]
```

### Softmax & ReLU
We had to create a softmax function to turn the output values in a probability table and a ReLU to normalize neuron activation, finishing or model 'structurewise'. 
It kind of worked pretty well. We're now in a very important step, because the only thing separating us from absolute success is figuring out how to train it. Right know it is only randomly guessing the output.

Have a look at our prediction function:
```rust
    fn apply_relu<const N: usize>(layer: &mut SVector<f64, N>) {
        for val in layer.iter_mut() {
            *val = val.max(0.0);
        }
    }

    fn apply_softmax<const N: usize>(layer: &SVector<f64, N>) -> SVector<f64, N> {
        let max_val = layer.max(); // for numerical stability
        let exps = layer.map(|x| (x - max_val).exp());
        let sum: f64 = exps.iter().sum();
        exps / sum
    }

    // It is good the image is not borrowed, since after prediction, the value can be dropped
    pub fn predict(&mut self, image: [f64; INPUT_SIZE]) -> SVector<f64, OUTPUT_SIZE> {
        self.input_layer = SVector::<f64, INPUT_SIZE>::from_row_slice(&image);
        self.hidden_layer_0 = self.weights_matrix_01 * self.input_layer;
        // Relu must be applied in every layer after firing neurons
        MLP::apply_relu(&mut self.hidden_layer_0);
        self.hidden_layer_1 = self.weights_matrix_12 * self.hidden_layer_0;
        MLP::apply_relu(&mut self.hidden_layer_1);
        self.output_layer = self.weights_matrix_23 * self.hidden_layer_1;
        MLP::apply_softmax(&self.output_layer)
    }
```
So after running our current `main.rs`:
```rust
fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let mut mlp = MLP::new();
    mlp.show_weights();
    mlp.randomize_weights();
    mlp.show_weights();
    let image: [u8; 784] = io::read_image()?;
    let normalized_image: [f64; 784] = parsers::normalize_image(image);
    let prediction = mlp.predict(normalized_image);
    println!("{:?}", prediction);
    io::render_output(&prediction);
    Ok(())
}
```
We finally get our fist guess in the form of a probabilities distribution, guessing 5.
The correct answer was 8 :(

```rust
[[2.539143790771743e-15, 2.412713529433148e-9, 2.422030173962812e-7, 0.0018935008110995345, 3.404185029212877e-8, 0.9161096634723647, 6.989882245023235e-5, 0.06949902885174457, 3.39904005601683e-11, 0.012427629350766809]]
 0 | 

 1 | 

 2 | 

 3 | 

 4 | 

 5 | ████████████████████████████████████████████████████████████████████████████████████████████

 6 | 

 7 | ███████

 8 | 

 9 | █
 ```