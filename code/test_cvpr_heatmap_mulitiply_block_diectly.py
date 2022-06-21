#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
import tensorflow_compression as tfc
import math
import time
import scipy.special

##new add
import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
from PIL import Image

from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
import tensorflow_probability as  tfp

tfd = tfp.distributions 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def get_image_size(input_file):
  I = Image.open(input_file)
  I_array = np.array(I)
  height_ori, width_ori, _ = np.shape(I_array)
  height = (height_ori // 64) * 64 if height_ori % 64 == 0 else (height_ori // 64 + 1) * 64
  width = (width_ori // 64) * 64 if width_ori % 64 == 0 else (width_ori // 64 + 1) * 64
  top_end = (height - height_ori) // 2
  left_end = (width - width_ori) // 2
  
  top_end = (height - height_ori) // 2
  left_end = (width - width_ori) // 2
  real_height_start = top_end
  real_height_end = top_end + height_ori
  real_width_start = left_end
  real_width_end = left_end + width_ori
  I_array_padded = np.zeros((1,height,width,3), np.uint8)
  I_array_padded[0,top_end:top_end+height_ori, left_end:left_end+width_ori,:]=I_array
  print('height_pad:', height, 'width_pad:', width)
  
  return I_array_padded,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)



def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image

def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image

def save_image(filename, image):
  """Saves an image to a PNG file."""

  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)
  
 
def residualblock(tensor, num_filters, scope="residual_block"):
  """Builds the residual block"""
  with tf.variable_scope(scope):
    with tf.variable_scope("conv0"):
      layer = tfc.SignalConv2D(
        num_filters//2, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(tensor)

    with tf.variable_scope("conv1"):
      layer = tfc.SignalConv2D(
        num_filters//2, (3,3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(output)

    with tf.variable_scope("conv2"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      output = layer(output)
      
    tensor = tensor + output
       
  return tensor


def residualblock_kenal(tensor, num_filters, kernel_size =3, scope="residual_block"):
  """Builds the residual block"""
  
  with tf.variable_scope(scope):
    with tf.variable_scope("conv0"):
      layer = tfc.SignalConv2D(
        num_filters//2, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(tensor)

    with tf.variable_scope("conv1"):
      layer = tfc.SignalConv2D(
        num_filters//2, (kernel_size,kernel_size), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(output)

    with tf.variable_scope("conv2"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      output = layer(output)
      
    tensor = tensor + output
       
  return tensor
  


def Multiply_resblock(tensor, num_filters, kernel_size =3, scope="multiply_residual_block"):
  """Builds the residual block"""
  with tf.variable_scope(scope):
    trunk_branch_3 = residualblock_kenal(tensor, num_filters, 3,scope="trunk_RB_3_3")
    trunk_branch_5 = residualblock_kenal(tensor, num_filters, 5,scope="trunk_RB_5_5")
    
    
    input_1 = tf.concat([trunk_branch_3, trunk_branch_5], axis=3)
    input_2 = tf.concat([trunk_branch_5, trunk_branch_3], axis=3)
    
    with tf.variable_scope("input_1_conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tfc.GDN(name='gdn'), name='signal_conv2d')
      input_1 = layer(input_1)
    
    
    with tf.variable_scope("input_2_conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tfc.GDN(name='gdn'), name='signal_conv2d')
      input_2 = layer(input_2)
    
    
    second_trunk_branch_3 = residualblock_kenal(input_1, num_filters, 3,scope="second_trunk_RB_3_3")
    second_trunk_branch_5 = residualblock_kenal(input_2, num_filters, 5,scope="second_trunk_RB_5_5")
    
    third_input = tf.concat([second_trunk_branch_3, second_trunk_branch_5], axis=3)
    
    with tf.variable_scope("conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      output = layer(third_input)
    tensor = tensor + output
  return tensor




def NonLocalAttentionBlock(input_x, num_filters, scope="NonLocalAttentionBlock"):
  """Builds the non-local attention block"""
  with tf.variable_scope(scope):
    trunk_branch = residualblock(input_x, num_filters, scope="trunk_RB_0")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_1")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_2")
    
    
    attention_branch = residualblock(input_x, num_filters, scope="attention_RB_0")
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_1")
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_2")

    with tf.variable_scope("conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      attention_branch = layer(attention_branch)
    attention_branch = tf.sigmoid(attention_branch)
  
  tensor = input_x + tf.multiply(attention_branch, trunk_branch)
  return tensor


def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  kernel_size = 3
  #Use three 3x3 filters to replace one 9x9
  
  with tf.variable_scope("analysis"):

    # Four down-sampling blocks
    for i in range(4):
      if i > 0:
        # with tf.variable_scope("Block_" + str(i) + "_layer_0"):
          # layer = tfc.SignalConv2D(
            # num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            # use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          # tensor2 = layer(tensor)

        # with tf.variable_scope("Block_" + str(i) + "_layer_1"):
          # layer = tfc.SignalConv2D(
              # num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              # use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          # tensor2 = layer(tensor2)
        
        # tensor = tensor + tensor2
        tensor = Multiply_resblock(tensor, num_filters, kernel_size =3, scope="multiply_residual_block"+str(i))


      if i < 3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):
          shortcut = tfc.SignalConv2D(num_filters, (1, 1), corr=True, strides_down=2, padding="same_zeros",
                                      use_bias=True, activation=None, name='signal_conv2d')
          shortcut_tensor = shortcut(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='gdn'), name='signal_conv2d')
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

        if i == 1:
          #Add one NLAM
          tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")
          

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
            use_bias=False, activation=None, name='signal_conv2d') 
          tensor_out = layer(tensor)
          
          
        with tf.variable_scope("Heatmap"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='gdn'), name='signal_conv2d') 
          tensor_heatin = layer(tensor)
          tensor_heatmap = residualblock(tensor_heatin,num_filters)
          tensor_heatmap = residualblock(tensor_heatmap,num_filters)
          tensor_heatmap = residualblock(tensor_heatmap,num_filters)
          tensor_heatmap = tensor_heatin + tensor_heatmap
          tensor_heatmap = tf.math.tanh(tensor_heatmap)
          tensor_heatmap = tf.nn.softsign(tensor_heatmap)

        #Add one NLAM
        tensor_out = NonLocalAttentionBlock(tensor_out, num_filters, scope="NLAB_1")
        tensor = tensor_heatmap*tensor_out

    return tensor




def hyper_analysis(tensor, num_filters):
  """Build the analysis transform in hyper"""

  with tf.variable_scope("hyper_analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters     
    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters 
    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  kernel_size = 3
  #Use four 3x3 filters to replace one 9x9
  
  with tf.variable_scope("synthesis"):

    # Four up-sampling blocks
    for i in range(4):
      if i == 0:
        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")

      if i == 2:
        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_1")
        
      # with tf.variable_scope("Block_" + str(i) + "_layer_0"):
        # layer = tfc.SignalConv2D(
          # num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          # use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
        # tensor2 = layer(tensor)

      # with tf.variable_scope("Block_" + str(i) + "_layer_1"):
        # layer = tfc.SignalConv2D(
          # num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          # use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
        # tensor2 = layer(tensor2)
        # tensor = tensor + tensor2
        tensor = Multiply_resblock(tensor, num_filters, kernel_size =3, scope="multiply_residual_block"+str(i))


      if i <3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):

          # Use Sub-Pixel to replace deconv.
          shortcut = tfc.SignalConv2D(num_filters*4, (1, 1), corr=False, strides_up=1, padding="same_zeros",
                                      use_bias=True, activation=None, name='signal_conv2d')
          shortcut_tensor = shortcut(tensor)
          shortcut_tensor = tf.depth_to_space(shortcut_tensor, 2)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):

          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(num_filters*4, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)         
          
        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='igdn', inverse=True), name='signal_conv2d')
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          
          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(12, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=None, name='signal_conv2d')
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)
          

    return tensor




def hyper_synthesis(tensor, num_filters):
  """Builds the hyper synthesis transform"""

  with tf.variable_scope("hyper_synthesis", reuse=tf.AUTO_REUSE):
    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters*2, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      tensor = layer(tensor)

    return tensor

def masked_conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    scope="masked"):
  
  with tf.variable_scope(scope):
    mask_type = mask_type.lower()
    batch_size, height, width, channel = inputs.get_shape().as_list()

    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
      "kernel height and width should be odd number"

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, weights_initializer, weights_regularizer)

    if mask_type is not None:
      mask = np.ones(
        (kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

      mask[center_h, center_w+1: ,: ,:] = 0.
      mask[center_h+1:, :, :, :] = 0.

      if mask_type == 'a':
        mask[center_h,center_w,:,:] = 0.

      weights *= tf.constant(mask, dtype=tf.float32)
      tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputs)

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    return outputs


  
def entropy_parameter(tensor, inputs, num_filters, training):
  """tensor: the output of hyper autoencoder (phi) to generate the mean and variance
     inputs: the variable needs to be encoded. (y)
  """
  with tf.variable_scope("entropy_parameter", reuse=tf.AUTO_REUSE):

    half = tf.constant(.5)

    if training:
      noise = tf.random_uniform(tf.shape(inputs), -half, half)
      values = tf.add_n([inputs, noise])
      
      

    else: #inference
      #if inputs is not None: #compress
      values = tf.round(inputs)
        

    masked = masked_conv2d(values, num_filters*2, [5, 5], "A", scope='masked')
    tensor = tf.concat([masked, tensor], axis=3)
      

    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          640, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          640, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters*9, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)


    #=========Gaussian Mixture Model=========
    prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
             tf.split(tensor, num_or_size_splits=9, axis = 3)
    scale0 = tf.abs(scale0)
    scale1 = tf.abs(scale1)
    scale2 = tf.abs(scale2)



    probs = tf.stack([prob0, prob1, prob2], axis=-1)
    probs = tf.nn.softmax(probs, axis=-1)
  
    # To merge them together
    means = tf.stack([mean0, mean1, mean2], axis=-1)
    variances = tf.stack([scale0, scale1, scale2], axis=-1)

    # =======================================
    ###cancel note
    #Calculate the likelihoods for inputs
    #if inputs is not None:
    if training:

      dist_0 = tfd.Normal(loc = mean0, scale = scale0, name='dist_0')
      dist_1 = tfd.Normal(loc = mean1, scale = scale1, name='dist_1')
      dist_2 = tfd.Normal(loc = mean2, scale = scale2, name='dist_2')

      #=========Gaussian Mixture Model=========
      likelihoods_0 = dist_0.cdf(values + half) - dist_0.cdf(values - half)
      likelihoods_1 = dist_1.cdf(values + half) - dist_1.cdf(values - half)
      likelihoods_2 = dist_2.cdf(values + half) - dist_2.cdf(values - half)

      likelihoods = probs[:,:,:,:,0]*likelihoods_0 + probs[:,:,:,:,1]*likelihoods_1 + probs[:,:,:,:,2]*likelihoods_2

      # =======REVISION: Robust version ==========
      edge_min = probs[:,:,:,:,0]*dist_0.cdf(values + half) + \
                 probs[:,:,:,:,1]*dist_1.cdf(values + half) + \
                 probs[:,:,:,:,2]*dist_2.cdf(values + half)
      
      edge_max = probs[:,:,:,:,0]* (1.0 - dist_0.cdf(values - half)) + \
                 probs[:,:,:,:,1]* (1.0 - dist_1.cdf(values - half)) + \
                 probs[:,:,:,:,2]* (1.0 - dist_2.cdf(values - half))
      likelihoods = tf.where(values < -254.5, edge_min, tf.where(values > 255.5, edge_max, likelihoods))

      
      likelihood_lower_bound = tf.constant(1e-6)
      likelihood_upper_bound = tf.constant(1.0)
      likelihoods = tf.minimum(tf.maximum(likelihoods, likelihood_lower_bound), likelihood_upper_bound)
      
    else:
      #values = None
      likelihoods = None

     ###added note
    #likelihoods = None
        
  return values, likelihoods, means, variances, probs






def compress(input, output, num_filters, checkpoint_dir):

    start = time.time()
    tf.set_random_seed(1)
    tf.reset_default_graph()
      
      
      #with tf.device('/cpu:0'):
        # Load input image and add batch dimension.
        
    #x = load_image(input)
    #print("x shape is {}".format(x.get_shape().as_list()))
    images_info = get_image_size(input)
    images_padded_numpy, size = images_info
    print("the size is {}".format(len(size)))
    real_height_start, real_height_end, real_width_start, real_width_end, height, width = size
    with tf.name_scope('Data'):
      images_padded = tf.placeholder(tf.float32, shape=(1, height, width, 3), name='images_ori')
      x = images_padded
      x_shape = tf.shape(x)

    y = analysis_transform(x, num_filters)

    # Build a hyper autoencoder
    z = hyper_analysis(y, num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    string = entropy_bottleneck.compress(z)
    string = tf.squeeze(string, axis=0)

    z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)

    # To decompress the z_tilde back to avoid the inconsistence error
    string_rec = tf.expand_dims(string, 0)
    z_tilde = entropy_bottleneck.decompress(string_rec, tf.shape(z)[1:], channels=num_filters)

    phi = hyper_synthesis(z_tilde, num_filters)


    # REVISION： for Gaussian Mixture Model (GMM), use window-based fast implementation    
    #y = tf.clip_by_value(y, -255, 256)
    y_hat = tf.round(y)


    #tiny_y = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters])
    #tiny_phi = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters*2]) 
    #_, _, y_means, y_variances, y_probs = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)
    #_, _, y_means, y_variances, y_probs, y_probs_lap, y_probs_log, y_probs_mix = entropy_parameter(phi, y, num_filters, training=False)
    _, _, y_means, y_variances, y_probs = entropy_parameter(phi, y, num_filters, training=False)
    
    x_hat = synthesis_transform(y_hat, num_filters)


    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    #x_hat = x_hat[0, :tf.shape(x)[1], :tf.shape(x)[2], :]

    #op = save_image('temp/temp.png', x_hat)

    # Mean squared error across pixels.
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)
    x_ori = x[:, real_height_start:real_height_end, real_width_start:real_width_end, :]
    x_hat = x_hat[:, real_height_start:real_height_end, real_width_start:real_width_end, :]
    mse = tf.reduce_mean(tf.squared_difference(x_ori, x_hat))


    with tf.Session() as sess:
      #print(tf.trainable_variables())
      sess.run(tf.global_variables_initializer())
      # Load the latest model checkpoint, get the compressed string and the tensor
      # shapes.
      #latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
      
      # latest = "models/model-1399000" #lambda = 14
        
      # print(latest)
      # tf.train.Saver().restore(sess, save_path=latest)
      
      
      vars_restore = [var for var in tf.global_variables()]
      saver_0 = tf.train.Saver(vars_restore)
      print(f'Loading learned model from checkpoint {checkpoint_dir}')
      saver_0.restore(sess, checkpoint_dir)

      
      #y_means, y_variances, y_probs
      y_means_values, y_variances_values, y_probs_values,string, x_shape, y_shape, num_pixels, y_hat_value, phi_value = \
              sess.run([y_means, y_variances, y_probs, string, tf.shape(x), tf.shape(y), num_pixels, y_hat, phi],feed_dict={images_padded:images_padded_numpy/255.0})
      

      
      minmax = np.maximum(abs(y_hat_value.max()), abs(y_hat_value.min()))
      minmax = int(np.maximum(minmax, 1))
      #num_symbols = int(2 * minmax + 3)
      print(minmax)
      #print(num_symbols)
      
      # Fast implementations by only encoding non-zero channels with 128/8 = 16bytes overhead
      flag = np.zeros(y_shape[3], dtype=np.int)
      
      for ch_idx in range(y_shape[3]):
        if np.sum(abs(y_hat_value[:, :,:, ch_idx])) > 0:
          flag[ch_idx] = 1

      non_zero_idx = np.squeeze(np.where(flag == 1))
      
      print("the zero numbers is {}".format(num_filters-len(non_zero_idx)))
      num = np.packbits(np.reshape(flag, [8, y_shape[3]//8]))

      # ============== encode the bits for z===========
      if os.path.exists(output):
        os.remove(output)

      fileobj = open(output, mode='wb')
      fileobj.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      fileobj.write(np.array([len(string), minmax], dtype=np.uint16).tobytes())
      fileobj.write(np.array(num, dtype=np.uint8).tobytes())
      fileobj.write(string)
      fileobj.close()



      # ============ encode the bits for y ==========
      print("INFO: start encoding y")
      encoder = RangeEncoder(output[:-4] + '.bin')
      samples = np.arange(0, minmax*2+1)
      TINY = 1e-10

       

      kernel_size = 5
      pad_size = (kernel_size - 1)//2
      
      
      
      padded_y = np.pad(y_hat_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))
      padded_phi = np.pad(phi_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))

      
      for h_idx in range(y_shape[1]):
        for w_idx in range(y_shape[2]):          

          
          extracted_y = padded_y[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]
          extracted_phi = padded_phi[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]

          
          # y_means_values, y_variances_values, y_probs_values = \
                          # sess.run([y_means, y_variances, y_probs], \
                                   # feed_dict={images_padded:images_padded_numpy/255.0, tiny_y: extracted_y, tiny_phi: extracted_phi})         
                                   
          # y_means_values, y_variances_values, y_probs_values, y_probs_lap_values, y_probs_log_values, y_probs_mix_values = \
                          # sess.run([y_means, y_variances, y_probs, y_probs_lap, y_probs_log, y_probs_mix], \
                                   # feed_dict={tiny_y: extracted_y, tiny_phi: extracted_phi})  

          
          
          for i in range(len(non_zero_idx)):
            ch_idx = non_zero_idx[i]
            
            # mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
            # sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
            # weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]
            
            # mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
            # sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
            # weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]
            # weight_lap = y_probs_lap_values[0, pad_size, pad_size, ch_idx, :]
            # weight_log = y_probs_log_values[0, pad_size, pad_size, ch_idx, :]
            # weight_mix = y_probs_mix_values[0, pad_size, pad_size, ch_idx, :]
            
            mu = y_means_values[0, h_idx, w_idx, ch_idx, :] + minmax
            sigma = y_variances_values[0, h_idx, w_idx, ch_idx, :]
            weight = y_probs_values[0, h_idx, w_idx, ch_idx, :]
            # weight_lap = y_probs_lap_values[0, h_idx, w_idx, ch_idx, :]
            # weight_log = y_probs_log_values[0, h_idx, w_idx, ch_idx, :]
            # weight_mix = y_probs_mix_values[0, h_idx, w_idx, ch_idx, :]

            start00 = time.time()

            # Calculate the pmf/cdf            
            pmf = (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                  (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] +\
                  (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]

            '''
            # Add the tail mass
            pmf[0] += 0.5 * (1 + scipy.special.erf(( -0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) * weight[0] + \
                      0.5 * (1 + scipy.special.erf(( -0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) * weight[1] + \
                      0.5 * (1 + scipy.special.erf(( -0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) * weight[2]
                      
            pmf[-1] += (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                       (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] + \
                       (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]
            '''
            
            # To avoid the zero-probability            
            pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
            pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
            cdf = list(np.add.accumulate(pmf_clip))
            cdf = [0] + [int(i) for i in cdf]
                      
            symbol = np.int(y_hat_value[0, h_idx, w_idx, ch_idx] + minmax )
            encoder.encode([symbol], cdf)


            

      encoder.close()

      size_real = os.path.getsize(output) + os.path.getsize(output[:-4] + '.bin')
      
      bpp_real = (os.path.getsize(output) + os.path.getsize(output[:-4] + '.bin'))* 8 / num_pixels
      bpp_side = (os.path.getsize(output))* 8 / num_pixels
      

      end = time.time()
      print("Time : {:0.3f}".format(end-start))

      psnr = sess.run(tf.image.psnr(x_hat, x_ori*255, 255),feed_dict={images_padded:images_padded_numpy/255.0})
      msssim = sess.run(tf.image.ssim_multiscale(x_hat, x_ori*255, 255),feed_dict={images_padded:images_padded_numpy/255.0})
      
      print("Actual bits per pixel for this image: {:0.4}".format(bpp_real))
      print("Side bits per pixel for z: {:0.4}".format(bpp_side))
      print("PSNR (dB) : {:0.4}".format(psnr[0]))
      print("MS-SSIM : {:0.4}".format(msssim[0]))
      
      
      return bpp_real, (end-start), psnr[0], msssim[0]


def overall_performance(metrics_list):
  psnr_rgb_list = []
  psnr_y_list = []
  psnr_u_list = []
  psnr_v_list = []
  msssim_rgb_list = []
  msssim_y_list = []
  bpp_list = []
  for metrics_item in metrics_list:
    psnr_rgb_list.append(metrics_item[0])
    psnr_y_list.append(metrics_item[1][0])
    psnr_u_list.append(metrics_item[1][1])
    psnr_v_list.append(metrics_item[1][2])
    msssim_rgb_list.append(metrics_item[2])
    msssim_y_list.append(metrics_item[3])
    bpp_list.append(metrics_item[4])
  bpp_avg = np.mean(bpp_list)
  RGB_MSE_avg = np.mean([255. ** 2 / pow(10, PSNR / 10) for PSNR in psnr_rgb_list])
  RGB_PSNR_avg = 10 * np.log10(255. ** 2 / RGB_MSE_avg)
  Y_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_y_list])
  Y_PSNR_avg = 10 * np.log10(255 ** 2 / Y_MSE_avg)
  U_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_u_list])
  U_PSNR_avg = 10 * np.log10(255 ** 2 / U_MSE_avg)
  V_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_v_list])
  V_PSNR_avg = 10 * np.log10(255 ** 2 / V_MSE_avg)
  yuv_psnr_avg = 6.0/8.0*Y_PSNR_avg + 1.0/8.0*U_PSNR_avg + 1.0/8.0*V_PSNR_avg
  msssim_rgb_avg = np.mean(msssim_rgb_list)
  msssim_y_avg = np.mean(msssim_y_list)

  print("overall performance")
  print("RGB PSNR (dB): {:0.2f}".format(RGB_PSNR_avg))
  print("YUV444 PSNR (dB): {:0.2f}".format(yuv_psnr_avg))
  print("RGB Multiscale SSIM: {:0.4f}".format(msssim_rgb_avg))
  print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_rgb_avg)))
  print("Y Multiscale SSIM: {:0.4f}".format(msssim_y_avg))
  print("Y Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_y_avg)))
  print("Actual bits per pixel: {:0.4f}\n".format(bpp_avg))



def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument("--autoregressive", "-AR", action="store_true", help="Include autoregressive model for training")
  parser.add_argument("--num_filters", type=int, default=192, help="Number of filters per layer.")
  parser.add_argument("--restore_path", default=None, help="Directory where to load model checkpoints.")
  parser.add_argument("--checkpoint_dir", default="train", help="Directory where to save/load model checkpoints.")
  parser.add_argument("--if_weight", type=int, default=0.0, help="weights")
  subparsers = parser.add_subparsers(title="commands", dest="command",
      help="commands: 'train' loads training data and trains (or continues "
           "to train) a new model. 'encode' reads an image file (lossless "
           "PNG format) and writes a encoded binary file. 'decode' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Trains (or continues to train) a new model.")
  train_cmd.add_argument("--train_root_dir", default="images", help="The root directory of training data, which contains a list of RGB images in PNG format.")
  train_cmd.add_argument("--batchsize", type=int, default=8, help="Batch size for training.")
  train_cmd.add_argument("--patchsize", type=int, default=256, help="Size of image patches for training.")
  train_cmd.add_argument("--lossWeight", type=float, default=0, dest="lossWeight", help="Weight for MSE-SSIM tradeoff.")
  train_cmd.add_argument("--lambda", type=float, default=0.01, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument("--last_step", type=int, default=1500000, help="Train up to this number of steps.")
  train_cmd.add_argument("--lr", type=float, default = 1e-4, help="Learning rate [1e-4].")
  train_cmd.add_argument("--lr_scheduling", "-lr_sch", action="store_true", help="Enable learning rate scheduling, [enabled] as default")
  train_cmd.add_argument("--preprocess_threads", type=int, default=16, help="Number of CPU threads to use for parallel decoding of training images.")

  # 'encode' subcommand.
  encode_cmd = subparsers.add_parser("encode", formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Reads a PNG file, encode it, and writes a 'bitstream' file.")
  # 'decode' subcommand.
  decode_cmd = subparsers.add_parser("decode",formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Reads a 'bitstream' file, reconstructs the image, and writes back a PNG file.")

  # Arguments for both 'encode' and 'decode'.
  for cmd, ext in ((encode_cmd, ".bitstream"), (decode_cmd, ".png")):
    cmd.add_argument("input_file", help="Input filename.")
    cmd.add_argument("output_file", nargs="?", help="Output filename (optional). If not provided, appends '{}' to the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args

def main(args):
  # Invoke subcommand.
  #os.environ['CUDA_VISIBLE_DEVICES'] = "2"
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
  if args.command == "train":
    train(args)
  elif args.command == "encode": # encoding
    if not args.output_file:
      args.output_file = args.input_file + ".bitstream"
    if os.path.isdir(args.input_file):
      dirs = os.listdir(args.input_file)
      test_files = []
      for dir in dirs:
        path = os.path.join(args.input_file, dir)
        if os.path.isdir(path):
          test_files += glob.glob(path + '/*.png')[:6]
        if os.path.isfile(path):
          test_files.append(path)
      if not test_files:
        raise RuntimeError(
          "No testing images found with glob '{}'.".format(args.input_file))
      print("Number of images for testing:", len(test_files))
      metrics_list=[]
      for file_idx in range(len(test_files)):
        file = test_files[file_idx]
        print(str(file_idx)+" testing image:", file)
        args.input_file = file
        #file_name = file.split('/')[-1]
        #args.output_file = args.output_file + file_name.replace('.png', '.bitstream')
        image_padded, size = get_image_size(args.input_file)
        metrics = encode(args, image_padded, size)
        metrics_list.append(metrics)
      overall_performance(metrics_list)
    else:
      image_padded, size = get_image_size(args.input_file)
      metrics = encode(args, image_padded, size, True)
  elif args.command == "decode": # decoding
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decode(args)

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)


