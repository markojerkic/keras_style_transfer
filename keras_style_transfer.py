# This code is written for Tensorflow 1.8
import time
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.contrib.keras.api.keras.applications import vgg19
from tensorflow.contrib.keras.api.keras.preprocessing.image import load_img, img_to_array

# Define constants
CONTENT_IMG_PATH = '/images/content.jpg'
STYLE_IMG_PATH = '/images/style.jpg'
OUTPUT_PATH = '/output/gen_img.jpg'
# Number of iterations to run
ITER = 10
# Weights of losses
CONTENT_WEIGHT = 0.01
STYLE_WEIGHT = 1.0
TV_WEIGHT = 1.0

# Define the shape of the output image
h, w = load_img(CONTENT_IMG_PATH).size
img_h = 400
img_w = int(h * img_h / w)


def preprocess(img_path):
    # Preprocessing to make the style transfer
    # possible
    img = load_img(img_path)
    img = img_to_array(img)
    # This dimensions are for Tensorflow backend
    img = imresize(img, (img_h, img_w, 3))
    img = img.astype('float64')
    # Add the batch dimension
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(img):
    if K.image_data_format() == 'channels_first':
        # For Theano
        img = img.reshape((3, img_h, img_w))
        img = img.transpose((1, 2, 0))
    else:
        img = img.reshape((img_h, img_w, 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


# Create Keras variables of input images
content_img = K.variable(preprocess(CONTENT_IMG_PATH))
style_img = K.variable(preprocess(STYLE_IMG_PATH))

if K.image_data_format() == 'channels_first':
    gen_img = K.placeholder(shape=(1, 3, img_h, img_w))
else:
    gen_img = K.placeholder(shape=(1, img_h, img_w, 3))

# Create a single tensor containing all three images
input_tensor = K.concatenate([content_img, style_img, gen_img], axis=0)

# Create a vgg19 model by running the input tensor though the vgg19 convolutional
# neural network, excluding the fully connected layers
model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
print('Model loaded')

# Create an output dictionary
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # Dot product of the flattened feature map and the transpose of the
    # flattened feature map
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, gen):
    assert K.ndim(style) == 3
    assert K.ndim(gen) == 3
    S = gram_matrix(style)
    G = gram_matrix(gen)
    channels = 3
    size = img_h * img_w
    # Euclidean distance of the gram matrices multiplied by the constant
    return K.sum(K.square(S - G)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(content, gen):
    assert K.ndim(content) == 3
    assert K.ndim(gen) == 3
    # Euclidean distance
    return K.sum(K.square(gen - content))


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_h - 1, :img_w - 1] - x[:, :, 1:, :img_w - 1])
        b = K.square(x[:, :, :img_h - 1, :img_w - 1] - x[:, :, :img_h - 1, 1:])
    else:
        # Move the image pixel by pixel, and calculate the variance
        a = K.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, 1:, :img_w - 1, :])
        b = K.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, :img_h - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


loss = 0.
# For content loss, we're using the 2nd convolutional layer form
# the 5th block
layer_features = outputs_dict['block5_conv2']
content_img_features = layer_features[0, :, :, :]
gen_img_features = layer_features[2, :, :, :]
loss += CONTENT_WEIGHT * content_loss(content_img_features, gen_img_features)

feature_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for name in feature_layer_names:
    layer_features = outputs_dict[name]
    style_features = layer_features[1, :, :, :]
    gen_img_features = layer_features[2, :, :, :]
    s1 = style_loss(style_features, gen_img_features)
    # We need to devide the loss by the number of layers that we take into account
    loss += (STYLE_WEIGHT / len(feature_layer_names)) * s1
loss += TV_WEIGHT * total_variation_loss(gen_img)

# Calculate gradients
grads = K.gradients(loss, gen_img)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

# Define a Keras function
f_output = K.function([gen_img], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_h, img_w))
    else:
        x = x.reshape((1, img_h, img_w, 3))
    # Update the loss and the gradients
    outs = f_output([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_value = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_value = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grads_value = grad_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grads_values = np.copy(self.grads_value)
        self.loss_value = None
        self.grads_value = None
        return grads_values


evaluator = Evaluator()

# Run L-BFGS optimizer
x = preprocess(CONTENT_IMG_PATH)

for i in range(ITER):
    print('Step {}'.format(i))
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxiter=300)
    print('    loss: {}'.format(min_val))
    # Save img
    img = deprocess_image(x)
    imsave('/output/img{}.jpg'.format(i), img)
    print('     Image saved. Time: {}'.format(time.time() - start_time))
