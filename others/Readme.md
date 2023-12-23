## Customize YOHO according to your needs
Here we describe how to customize YOHO to apply the group processing pipeline according to different levels of requirements, including direct generalization, retraining on other datasets and backbone replacement.
### You have a point cloud file(.ply/.pcd), and you want to extract FCGF or YOHO features.
  I provide two quite simple scripts in ```simple_yoho``` but I have not fully checked.
  - ```fcgf_feat.py``` can be used for FCGF feature extraction, the output is a set of down sampled points and their FCGF features.
  - ```yoho_extract.py``` can be used for YOHO feature extraction, the output is 5000 randomly sampled keypoints and their corresponding yoho features.
  - **NOTE:** a key parameter you should carefully set for both algos is ```voxel_size```:
    - voxel_size = 0.025*(a rough scale of your pcs)/3m, the explanation is in the following context. (For indoor scene, just setting it to 0.025 is always ok.)
    - voxel_size should be set to the same for the source and target pcs.

### Direct generalization on other testsets.
  Here we utilize the pre-trained model(FCGF backbone) in the repo directly on a individual testset, namely, ```DS```:
  - Prepare ```DS``` following the data structure as:
    ```
    data/
    ├── origin_data/
        ├── DS/
          └── scene0/
                ├── PointCloud/
                  ├── cloud_bin_0.ply
                  ├── gt.log          #scene pairs used, according to http://redwood-data.org/indoor/registration.html
                  └── gt.info         #needed for RR calculation, according to http://redwood-data.org/indoor/registration.html
                └── Keypoints/
                  └── cloud_bin_0Keypoints.txt    #indexes in cloud_bin_0.ply of keypoints
    ```
    **Note:**
    - The cloud_bin_0Keypoints.txt contains the point indexes of 5000 randomly sampled points of cloud_bin_0.ply.
    - gt.log and gt.info files are used for evaluation and require given ground truth tranformations, if you only have a set of point clouds, just use the fake ones in ```./others```.
  - Logging ```DS``` into ```utils/dataset-->get_dataset_name()``` by adding codes following:
    ```
        if dataset_name=='DS':
          datasets={}
          datasets['wholesetname']=f'{DS}'
          scenes=["scene0","scene1"]
          stationnums=[pcnum0,pcnum1]
    
          for i in range(len(scenes)):
              root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
              datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
              datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
          return datasets
    ```
  - Assume the scale difference ```s``` between ```DS``` and 3DMatch. FCGF is a spatial-voxelization encoder and we set voxel size to 0.025m for 3DMatch in training, thus the voxelized ```DS``` should have an similar scale by voxel size ```a~0.025*s```, which will be applied in preparing the testset as:
    ```
    python YOHO_testset.py --dataset DS --voxel_size a
    ```

  - Afterwards, YOHO can be applied similarly as mention above:
    ```
    python Test.py --Part p  --max_iter m --dataset DS --ransac_d r --tau_2 i --tau_3 c
    ```
    where ```p``` is PartI or PartII, ```m``` is the execuation times of RANSAC, ```DS``` is the name of testset, ```r``` is the inlier threshold used in RANSAC, ```i``` is the inlier threshold for FMR and IR calculation, ```c``` is the success threshold for RR calculation.

### Retrain the FCGF backbone YOHO.
  Here we utilize FCGF backbone and train with new dataset ```DS_train```:
  - Retrain the FCGF with  ```DS_train``` according to [FCGF](https://github.com/chrischoy/FCGF) but limit the random rotation range in trainset to less than 50deg.
  - Replace the model in ./model/Backbone to your owns.
  - Prepare the ```DS_train``` as the data structure above and log it to ```utils/dataset-->get_dataset_name()``` same as above and identify the scenes for valization(same as ```if name=='3dmatch_train'```).
  - Deside the voxel size ```v``` for FCGF backbone and prepare the trainset for YOHO.
  - Execute the train process as the command in the main repo. 


### Use other backbones.
  - For fully-conv ones(like FCGF), we rotate the point cloud according to the group and extrate features on them, then get the group feature for the predefined keypoints by NN search.
  - For patch-based ones(like PointNet, smoothnet), we extract the local patch for keypoints and rotate them according to the group for group feature extraction.
  - Train the backbone, rewrite the ```YOHO_trainset.py``` and ```YOHO_testset.py``` for using it. Other commands will be the same.