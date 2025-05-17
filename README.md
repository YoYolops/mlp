# mlp
Today is may 16th, 2025. I'll TRY to build a multilayer perceptron from the ground up with rust. This repository will only go public if i succeed :)

### THE PLAN 
I will use the MNIST dataset to train my perceptron to be able to recognize handwritten digits. What can go wrong?

To my surprise, the MNIST dataset comes in an alien format. After some research it appears to be provided as raw binary
big-endian integers (?). Well, a first challenge so soon wasn't expected. 

Here, my plan ruins. The original idea (i omitted this part) was to build EVERYTHING from scratch, which i've just gave
up, or the likelyhood of me ending this project would be too low.

The thing is: I will use some library to handle this image reading 

¯\\_(ツ)_/¯

### Reading The Images
After reading [a stack overflow post](https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format) and asking some questions to gpt, i have kinda figured out how to read the images:

Offset | Meaning
-------|----------------------------
0–3    | Magic number      (2051)
4–7    | Number of images  (60000)
8–11   | Number of rows    (28)
12–15  | Number of columns (28)

*the offset is in bytes

