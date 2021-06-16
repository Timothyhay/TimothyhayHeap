---
layout: blogpage
title: AutoEncoder on earth
comments: true
tags: Deep-Learning
---

Each convolution layer consists of several convolution channels (aka. depth or filters). In practice, they are a number such as 64, 128, 256, 512 etc. This is equal to number of channels in the output of a convolutional layer. kernel_size, on the other hand, is the size of these convolution filters. In practice, they take values such as 3x3 or 1x1 or 5x5. To abbreviate, they can be written as 1 or 3 or 5 as they are mostly square in practice.

Edit

Following quote should make it more clear.

Discussion on vlfeat

Suppose X is an input with size W x H x D x N (where N is the size of the batch) to a convolutional layer containing filter F (with size FW x FH x FD x K) in a network.

The number of feature channels D is the third dimension of the input X here (for example, this is typically 3 at the first input to the network if the input consists of colour images). The number of filters K is the fourth dimension of F. The two concepts are closely linked because if the number of filters in a layer is K, it produces an output with K feature channels. So the input to the next layer will have K feature channels.

The FW x FH above is filter size you are looking for.

Added

You should be familiar with filters. You can consider each filter to be responsible for extracting some type of feature from a raw image. The CNNs try to learn such filters i.e. the filters parametrized in CNNs are learned during training of CNNs. You apply each filter in a Conv2D to each input channel and combine these to get output channels. So, the number of filters and the number of output channels are the same.




0


1
I am attempting to build my first Autoencoder neural net using TensorFlow. The dimensions of the layers in the encoder and decoder are the same, just reversed. The autoencoder learns to compress and reconstruct image data to a reasonable standard, but I would like to try to improve its performance by instead having the decoder as the exact transpose of the encoder.

I am lost with how to do this in TensorFlow.

Here is a snippet of the construction of my network:

imgW, imgH = 28, 28
encoderDims = [
    imgW * imgH,
    (imgW // 2) * (imgH // 2),
    (imgW // 3) * (imgH // 3),
    (imgW // 4) * (imgH // 4)
]
decoderDims = list(reversed(encoderDims))

encoderWeights, encoderBiases = [], []
decoderWeights, decoderBiases = [], []
for layer in range(len(encoderDims) - 1):
    encoderWeights.append(
        tf.Variable(tf.random_normal([encoderDims[layer], encoderDims[layer + 1]]))
    )
    encoderBiases.append(
        tf.Variable(tf.random_normal([encoderDims[layer + 1]]))
    )
    decoderWeights.append(
        tf.Variable(tf.random_normal([decoderDims[layer], decoderDims[layer + 1]]))
    )
    decoderBiases.append(
        tf.Variable(tf.random_normal([decoderDims[layer + 1]]))
    )

input = tf.placeholder(tf.float32, [None, imgW * imgH])
encoded = input
for layer in range(len(encoderDims) - 1):
    encoded = tf.add(tf.matmul(encoded, encoderWeights[layer]), encoderBiases[layer])
    encoded = tf.nn.sigmoid(encoded)

decoded = encoded
for layer in range(len(decoderDims) - 1):
    decoded = tf.add(tf.matmul(decoded, decoderWeights[layer]), decoderBiases[layer])
    if layer != len(decoderDims) - 2:
        decoded = tf.nn.sigmoid(decoded)

loss = tf.losses.mean_squared_error(labels=input, predictions=decoded)
train = tf.train.AdamOptimizer(learningRate).minimize(loss)
The two issues I do not know how to overcome are:

How can I adjust only the encoder parameters during training with respect to the loss?
How can I create the decoder weights and biases in such a way that after each iteration of training of the encoder parameters, they are set as the transpose of the newly adjusted encoder parameters?


I doubt that this will outperform regular autoencoders. But if you should get surprisingly good results, let the community know. Regarding your questions:

1.) As you have to use some reconstruction error between input and output, the only option is to train with the whole network (e.g. encoder and decoder as a whole). But, you could set a flag for the decoder variables, that prevents them from getting altered by the algorithm after getting initialized. Set them to trainable=False. After training for an epoch, you can set them manually to the transposed encoder weights.

2.) Here, I am not sure how you interpret 'transpose'. If you mean that the weights of layer 1 of the encoder should match those of the last layer of the decoder, you can try this:

for layer in range(len(encoderWeights)):
    decoderWeights[-layer-1] = tf.transpose(encoderWeights[layer])
If you want to transpose the layer matrices individually, you can use said tf.tranpose(). From a mathematical standpoint, the correct reverse operation of a matrix multiplication would be the inverse matrix, if it is defined. TensorFlow does provide tf.matrix_inverse() for this. However, be very careful when using this as a reasonable result is not guaranteed.




https://stackoverflow.com/questions/51185093/how-to-create-an-autoencoder-where-the-encoder-decoder-weights-are-mirrored-tra