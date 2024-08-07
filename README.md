# Scratch Implementation of GenAI architectures like GAN

# TODO

- [ ] [VITGAN: TRAINING GANS WITH VISION TRANSFORMERS](https://openreview.net/pdf?id=dwg5rXg1WS_)
- [ ] [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/pdf/2102.07074.pdf)
- [ ] [StyleSwin: Transformer-based GAN for High-resolution Image Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_StyleSwin_Transformer-Based_GAN_for_High-Resolution_Image_Generation_CVPR_2022_paper.pdf)
- [x] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)
- [ ] [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf)
- [ ] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
- [x] [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)
- [ ] [A Style-Based Generator Architecture for Generative Adversarial Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf)

# Helpful Papers 

- [A Large-Scale Study on Regularization and Normalization in GANs](https://arxiv.org/pdf/1807.04720)
- [Least Squares Generative Adversarial Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf)
- [CONSISTENCY REGULARIZATION FOR GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1910.12027)
- [Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318)
- [Wasserstein GAN](https://arxiv.org/pdf/1701.07875)

# GAN Hacks

- No sparse gradients (no maxpool, ReLU)
- Slow learning rate 
- use batch normalization, LeakyReLU, dropout
- Implement discriminator loss function in two goes 