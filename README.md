This is the project page for the paper:

>[**Architecture Disentanglement for Deep Neural Networks**](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Architecture_Disentanglement_for_Deep_Neural_Networks_ICCV_2021_paper.pdf),  
> Jie Hu, Liujuan Cao, Tong Tong, Ye Qixiang, ShengChuan Zhang, Ke Li, Feiyue Huang, Ling Shao, Rongrong Ji

## Updates
- (2021.11.18) The project page for NAD is avaliable.

## Pretrained Models For Place
ImageNet pretrained model can be downloaded online. As for the place dataset, we trained the four networks on place365 dataset.
The pretrained model can be download at [google driver](https://drive.google.com/drive/folders/1IVf-5kgncni1c3F4cmGQA9nn3idQYtgc?usp=sharing), they should be placed at the folder `NAD/pretrain_model/`

#### Requirements
- Python=3.7
- PyTorch=1.7.1, torchvision=0.8.2, cudatoolkit=10.1

#### Steps (vgg16 and imagenet for example)
1. Install Anaconda, create a virtual environment and install the requirements above. And then
```
git clone https://github.com/hujiecpp/NAD
```
2. Download ImageNet dataset and Place365 dataset and then modify the `NAD/tools/config.py`. As for the Place365 dataset, use 'NAD/tools/make_dataset.py' to convert it to a suitable format.

3. Find the path for all categories at network
```
CUDA_VISIBLE_DEVICES=0 python findpath.py --net vgg16 --dataset imagenet --beta 4.5
```

4. Test one image using its path
```
CUDA_VISIBLE_DEVICES=0 python cam_1x1.py --model vgg16 --dataset imagenet --epoch 20 --mask_rate 0.05
```

5. Generate 2x2 images randomly and test path hit rate for each layer
```
CUDA_VISIBLE_DEVICES=0 python cam_2x2.py --model vgg16 --dataset imagenet --epoch 20 --mask_rate 0.05
```

6. Calculate the top3 substructure similarity for each class and compare it with the result of top3 classified by the classification network
```
CUDA_VISIBLE_DEVICES=0 python similarSubArch.py --model vgg16 --dataset imagenet --epoch 20
```

## Citation

If our paper helps your research, please cite it in your publications:

```BibTeX
@inproceedings{hu2021architecture,
  title={Architecture disentanglement for deep neural networks},
  author={Hu, Jie and Cao, Liujuan and Tong, Tong and Ye, Qixiang and Zhang, Shengchuan and Li, Ke and Huang, Feiyue and Shao, Ling and Ji, Rongrong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={672--681},
  year={2021}
}
```
