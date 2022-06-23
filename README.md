# Asymmetric-Learned-Image-Compression-with-MRSB-IM-and-PQF


This repository contains the code for reproducing the results with trained models, in the following paper:

Our code is based on the paper named Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. [arXiv](https://arxiv.org/abs/2001.01568), CVPR2020. Zhengxue Cheng, Heming Sun, Masaru Takeuchi, Jiro Katto

Our paper is Asymmetric Learned Image Compression with Multi-Scale Residual Block, Importance Map, and Post-Quantization Filtering. [arXiv](https://arxiv.org/abs/2206.10618).
Haisheng Fu, Feng Liang, Jie Liang, Binglin Li, Guohe Zhang, Jingning Han



## Paper Summary

Recently, deep learning-based image compression has made signifcant progresses, and has achieved better ratedistortion (R-D) performance than the latest traditional method, H.266/VVC, in both subjective metric and the more challenging objective metric. However, a major problem is that many leading learned schemes cannot maintain a good trade-off between performance and complexity. In this paper, we propose an effcient and effective image coding framework, which achieves similar R-D performance with lower complexity than the state of the art. First, we develop an improved multi-scale residual block (MSRB) that can expand the receptive feld and is easier to
obtain global information. It can further capture and reduce the spatial correlation of the latent representations. Second, a more advanced importance map network is introduced to adaptively allocate bits to different regions of the image. Third, we apply a 2D post-quantization flter (PQF) to reduce the quantization error, motivated by the Sample Adaptive Offset (SAO) flter in video coding. Moreover, We fnd that the complexity of encoder and decoder have different effects on image compression performance. Based on this observation, we design an asymmetric paradigm, in which the encoder employs three stages of MSRBs to improve the learning capacity, whereas the decoder only needs one stage of MSRB to yield satisfactory reconstruction, thereby reducing the decoding complexity without sacrifcing
performance. Experimental results show that compared to the state-of-the-art method, the encoding and decoding time of the proposed method are about 17 times faster, and the R-D performance is only reduced by less than 1% on both Kodak and Tecnick datasets, which is still better than H.266/VVC(4:4:4) and other recent learning-based methods. 

### Environment 

* Python==3.6.4

* Tensorflow==1.14.0

* [RangeCoder](https://github.com/lucastheis/rangecoder)

```   
    pip3 install range-coder
```

* [Tensorflow-Compression](https://github.com/tensorflow/compression) ==1.2

```
    pip3 install tensorflow-compression or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
```

### Test Usage

* Download the pre-trained [models](https://pan.baidu.com/s/1VZ8EZZzX8VKJg4auKxVytQ) (The Extraction code is i6p3. These models are optimized by PSNR using lambda = 0.0016(number filters=128)) and lambda = 0.03(number filters=256)).

* Run the following py files can encode or decode the input file. 

```
   python Encoder_Decoder_cvpr_blocks_leaky_GLLMM_directly_bits_github.py
   note that:
   endcoder_main(); // the Encoder code
   decoder_main();  // the Decoder  code
   path ='xxx';     // the test image 
   save_image_name_path=''; // save the bit stream files.
   num_filters = 128 or 256;  // 128 for low bit rates and 256 for high bit rates.
   
```



## Evaluation Results
*The result on Kodak and Tecnick datasets are shown in [Results](https://github.com/fengyurenpingsheng/Asymmetric-Learned-Image-Compression-with-MRSB-IM-and-PQF/blob/main/images/)

## Notes


If you think it is useful for your reseach, please cite our paper. 
```
@misc{fuAsymmetric,
      title={Asymmetric Learned Image Compression with Multi-Scale Residual Block, Importance Map, and Post-Quantization Filtering}, 
      author={Haisheng Fu, Feng Liang, Jie Liang, Binglin Li, Guohe Zhang, Jingning Han},
      year={2022},
      eprint={2206.10618},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
