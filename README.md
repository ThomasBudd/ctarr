A deep learning-based tool for fast and robust cropping of anatomical regions on CT images.
For an in detail description of the algorithms please check our [paper](https://doi.org/10.59275/j.melba.2024-f5fc).

# Installation

The recommended way to install the library is to simply clone this repository and install via pip:

```
git clone https://github.com/ThomasBudd/ctarr
cd ctarr
pip install .
```

The code can be run on the CPU, but we recommend using a CUDA compatible GPU.

# Data management

The library is build to process (3d) CT images and expectes the volumes to be stored in the nifti format. If you're having dicom data we suggest the usage of a the usage of [this](https://github.com/icometrix/dicom2nifti) dicom to nifti converter. Nifti images can be stored as '.nii' or '.nii.gz' file and read by ctarr via the load_scan function:

```
from ctarr.utils.io import load_scan
scan = load_scan("\PATH\TO\NIFTI\IMAGE.nii.gz")
```

The 'scan' is simply a dictionary containing some information on the image. It will be used later to save intermediate results used for the cropping and undoing the cropping.

You don't have to follow any particular file or folder structure if you're only using per-computed bounding boxes. However, when computing new bounding boxes you need to store the dataset in a particular structure. To process such a dataset you have to build a folder structure similar to the one used in the medical decathlon challenges. A parent folder contains two folders 'images' and 'labels' (slightly different naming such as 'imagesTr' will also work as long as the two folders start with 'images' and 'labels') with the corresponding nifiti files. The two files of an image segmentation pair in the two folders must have the same name! Here is an example:

    PATH_TO_DATASET
      ├── images
      │   ├── case_001.nii.gz
      │   ├── case_002.nii.gz
      │   ├── case_003.nii.gz
      │   ├── ...
      ├── labels
      │   ├── case_001.nii.gz
      |   |── case_002.nii.gz
      │   ├── case_003.nii.gz
      │   ├── ...

You can create an itertable dataset object and load individual scans like this:

```
from ctarr.utils.io import dataset
ds = dataset(PATH_TO_DATASET)
scan = ds[0]
```
As an alternativ you can create a scan as a dictionary with at least the two keys 'image' and 'spacing'. The value of 'image' must be a numpy array with the 4d image tensor (first 1 channel, then z axis) and the value of 'spacing' a numpy array of length 3 with the voxel spacing of the three spatial axes (z first!) in mm.

# Cropping of precomputed anatomical regions
First we will have to create a cropping object for the desired anatomical region of interest with name NAME:

```
from ctarr import CTARR
ctarr = CTARR(NAME)
```

If you enter no name (NAME=None) the object will list the names of all available bounding boxes. Optionally you can increase the bounding boxes by an additional margin (as described in the paper), the default is additional_margin=10 (ie 1cm). CTARR also tries to infere the orientation of the scans as described in the paper. By default, flippings of the z axis and 90 degrees rotations in xy plane are considered. You can enable and disable this and flipping of the x and y axis with the input arguments allow_{z/x/y}_flipping and allow_xy_rotation. To run the cropping pipeline for the images (and segmentations if available) you can simply call the ctarr object:

```
ctarr(scan)
```

This will add a key 'image_crops' to the scan that contains list of the crop(s) of your bounding box(es). Keep in mind that you might have multiple bounding boxes for you anatomical region of interest, such as when considering both kidneys. In case the scan also contained a segmentation under the key 'label' the cropped segmentation will be available via the 'label_crops' key. If you want to crop any other array you can also use the ctarr.crop function directly. Both functions come with some optional arguments as listed below:

```
ctarr(scan,                 # as loaded by the ctarr io
      padding_val=None,     # (optional) use to pad the crop in case parts of the bounding box are outside of the image
      target_spacing=None,  # (optional) use to resize the crop to a desired voxel spacing
      z_only=False,   	    # (optional) to only crop in z direction
      mode='nearest')       # (optional) interpolation mode used in torch.nn.functional.interpolate
```

The padding_val argument can be helpful when the anatomical region is not fully contained in the image, ie. when parts of the bounding box is outside of the image. The default behaviour will ignore these parts of the bounding box and return only the part of inside the image. This means that an empty array will be returned when there is no intersection between the bounding box and the image. To pad the crop such that it contains the full bounding box, set the padding_val argument. Options are 'min' to use the minimum value of each channel, or a list of floats to use for each channel.

Deep learning-based processing pipelines of 3d images often resize the images to the same voxel spacing during preprocessing (such as nnUNet). This is integrated in the crop function by the target_spacing argument. If no padding is applied, setting the target_spacing argument will cause the resampling of the voxel size to the desired voxel size. When setting both a padding value and the target_spacing, the method will compute the size of the bounding box in the atlas coordinate system with the target_spacing and resize all crops to this size. This means that all crops in a dataset will have the same size, which is handy when dealing with 3d ResNet such as for 3d classification problems.

On last option the library holds is to only perform the cropping only in z direction and leave the xy plane as is. This can be helpful for example when separating the abdominal part from a whole body scan.

You can also apply the cropping to an entire dataset and store the results as niftis with a single line of code!

```
ctarr.crop_and_store_data(PATH_TO_DATASET, TARGET_PATH)
```

The cropped images will be stored at TARGET_PATH. You can also use the padding_val and target_spacing arguments in this function. By default the folder for the images will be called 'images' and same for the labels (if available). To add an extension use the ext='' argument (e.g. ext='Tr' --> 'imagesTr'). If the cropping results in an empty array because the image did not contain any parts of the bounding box, the array will not be stored. If you want to exclude images that show only very little context of the bounding box use the min_ax_len argument to set a threshold in mm.

If you work with medical image segmentation you can use these methods to create crops of images to train and evaluate your segmentation algorithms. But typically you will also want to bring the segmentation of the crop back to the full size and original orientation. In this situation you can use the method

```
arr = ctarr.undo_cropping(arr_crops,        # list of the crops obtained from the bounding box(es)
                          scan,             # corresponding scan
                          fill_val='min',   # constant value used for padding to original size
                          mode='nearest')   # interpolation mode in case the crops were resized
```

# Inference of own bounding boxes

If you have a dataset of CT images and segmentation of your objects of interest you can use the following function to compute bounding boxes for your problem:

```
ctarr.infere_bounding_boxes(PATH_TO_DATASET)
```

The following arguments can be optionally set. When your segmentations have multiple classes they will be converted to a binary mask by default. To use only a subset of classes use the 'used_classes' argument and set it to a list of integers of classes you want to use. The 'margin' argument will increase the bounding box by a given amount (in mm) in each direction after inference is done. The 'threshold' argument sets the threshold applied to the overlap of the segmentation masks. It makes sense to increase it when you labels are noisy. To disable to computation of multiple bounding boxes set allow_multiple_boxes=False. After finishing the computation of this new bounding box, you can save it via

```
ctarr.store_bounding_box(NAME)
```

to the cloned git repo at ctarr/file_storage. By default the method will not overwrite other boxes with the same name (overwrite=False). 

# Recommended usage

First, you should consider whether there exists a precomputed bounding box for your anatomical region of interest. If so check if the cropping with this box removes foreground voxels from your dataset. This can happen if the anatomical region of interest in enlarged in your cohort. If this is the case you can either increase the additional_margin, however we suggest to compute new bounding boxes for your problem.

When computing a new bounding box, you can check the overlap of the segmentation masks visually. I like to use "pseudo X-ray images" for this like suggested in this code snippit:

```
from ctarr import CTARR
import matplotlib.pyplot as plt

ctarr = CTARR
ovlp = ctarr.infere_bounding_boxes(PATH_TO_DATASET)

plt.subplot(121)
plt.imshow(ctarr.seg_atlas.sum((0, 2)).cpu().numpy(), cmap='gray')
plt.contour(ovlp > 0.01, colors='blue')
plt.subplot(122)
plt.imshow(ovlp.sum(1), cmap='hot')
```

If your goal is to use the cropping for image segmentation we recommend using more additional_margin when creating the training dataset compared to performing inference. Even though the cropping prevents false positives far away from the actual region of interest, it is still possible that some segmentation algorithms produce false positives at the edges of the crop. To familiarise the algorithms with structures occurring close to the edge of the crops it makes sense to increase the bounding box during training and use the original size during inference.

If you're interested in classification or regression instead we recommend using the padding_val and target_spacing argument of the crop function. This will ensure that all crops show the same anatomical region and are of the same spatial dimensions. In our experience this significantly simplifies the training and inference of approaches based on 3d Res-Nets or similar architectures.
