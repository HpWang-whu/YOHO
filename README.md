# You Only Match Once: Point Cloud Registration with Rotation-equivariant Descriptors

In this paper, we propose a novel local descriptor-based framework, called You Only Match Once (YOMO), for the registration of two unaligned point clouds. In contrast to most existing local descriptors which rely on a fragile local reference frame to gain rotation invariance, the proposed descriptor achieves the rotation invariance by recent technologies of group equivariant feature learning, which brings more robustness to point density and noise. Meanwhile, the descriptor in YOMO also has a rotation-equivariant part, which enables us to estimate the registration from just one correspondence.  Such property reduces the searching space for feasible transformations, thus greatly improves both the accuracy and the efficiency of YOMO. Extensive experiments show that YOMO achieves superior performances with much fewer needed RANSAC iterations on three widely-used datasets, the 3DMatch/3DLoMatch datasets and the ETH dataset. 

## News

- 2021.7.6 The code of the FCGF backbone YOMO is released.

## Performance and efficiency

(1)<img src="figures/performance.jpg" alt="Performance" style="zoom:50%;" />
(2)<img src="figures/Time.jpg" alt="Time" style="zoom:50%;" />

## Requirements

Here we offer the FCGF backbone YOMO, so the FCGF requirements need to be met:

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher

## Installation

Create the anaconda environment:

```
conda create -n fcgf_yomo python=3.7
conda activate fcgf_yomo
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch 
#We have checked pytorch1.7.1 and you can get the pytorch from https://pytorch.org/get-started/previous-versions/ accordingly.

#Install MinkowskiEngine, here we offer two ways according to the https://github.com/NVIDIA/MinkowskiEngine.git
(1) remove MinkowskiEngine here and pip install git+https://github.com/NVIDIA/MinkowskiEngine.git
(2) #Or use the MinkowskiEngine folder here.
    cd MinkowskiEngine
    conda install openblas-devel -c anaconda
    export CUDA_HOME=/usr/local/cuda-11.1 #We have checked cuda-11.1.
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    cd ..

pip install -r requirements.txt
```

KNN build:

```
cd knn_search/
export CUDA_HOME=/usr/local/cuda-11.1 #We have checked cuda-11.1.
python setup.py build_ext --inplace
cd ..
```



## Data Preparation

We need the 3DMatch dataset (Train, Test) and the 3DLoMatch dataset (Test).

We offer the origin train dataset containing the point clouds (.ply) and keypoints (.txt, 5000 per point cloud) here [TrainData](). With which, you can train the YOMO yourself.

We offer the origin test datasets containing the point clouds (.ply) and keypoints (.txt, 5000 per point cloud) here [TestData]() .

Please place the data to ```./data/origin_data``` for organizing the data structure as:

- data
  - origin_data
    -  3dmatch
        - sun3d-home_at-home_at_scan1_2013_jan_1
            - Keypoints
            - PointCloud
    - 3dmatch_train
      - bundlefusion-apt0
        - Keypoints
        - PointCloud

## Train

To train YOMO yourself, you need to prepare the origin trainset with the backbone FCGF. We have retrained the FCGF with the rotation argument in [0,50] deg and the backbone model is in ```./model/backbone```. With the TrainData downloaded above, you can create the YOMO trainset with:

```
python YOMO_trainset.py
```

Warning: the process above need 300G storage space.

Or you can directly download the trainset and validation set we pre process here [YOMOtrainset]() and you are free to download the TrainData above.

The training process of YOMO is two-stage, you can run which with the commands sequentially:

```
python Train.py --Part PartI
python Train.py --Part PartII
```

We also offer the pretrained models in ```./model/PartI_train``` and ```./model/PartII_train```.

## Demo

With the pretrained models, you can try the YOMO by:

```
python YOMO_testset.py --dataset demo
python Demo.py
```

## Test

With the TestData downloaded above, the test on 3DMatch and 3DLoMatch can be done by:

- Prepare testset

```
python YOMO_testset.py --dataset 3dmatch
```

- Eval the results or diectly download the group descriptors here [Descriptors]():

```
python Test.py --Part PartI --max_iter 1000 --dataset 3dmatch
```

where PartI is yomo-c and PartII is yomo-o, max_iter is the ransac times and dataset can be replaced as 3dLomatch.



## Related Projects

We thanks greatly for the FCGF, PerfectMatch and Predator for the backbone and datasets.

- [FCGF](https://github.com/chrischoy/FCGF)
- [3DSmoothNet](https://github.com/zgojcic/3DSmoothNet) 
- [Predator](https://github.com/overlappredator/OverlapPredator) 

