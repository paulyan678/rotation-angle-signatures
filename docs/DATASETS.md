# Dataset setup

Dataset files are excluded from Git. Set `data_root` in the YAML configuration or use the
default `data/` directory. The classification experiment uses images only; original class
labels are not used in the contrastive or binary-pair task.

Set `data_revision` to a stable identifier for the exact local dataset release/manifests.
Change it whenever files, subsets, or preprocessing inputs change; it participates in every
job ID and prevents stale-result reuse. The repository cannot infer official release IDs
for manually licensed datasets that the paper does not identify.

## Automatically downloaded by torchvision

The adapters request official torchvision downloads for:

- CIFAR-10
- MNIST
- Fashion-MNIST
- SVHN
- EuroSAT
- Caltech-101
- Flowers-102
- Food-101

iNaturalist 2018 must be downloaded manually because its size and terms make silent
download inappropriate; its images are scanned recursively because class labels are not
used by this experiment.

## Image-folder datasets

The following classification datasets are scanned recursively for PNG, JPEG, TIFF, or BMP
images. They may either put files directly below the named root or in an `images/` folder.

```text
data/
  tiny_imagenet/
  stanford_dogs/
  plant_village/
  chest_xray14/
  inaturalist/
```

For exact work, acquire the versions cited by the paper and preserve their official splits
outside this repository. The paper does not provide hashes, subset lists, or a precise
iNaturalist version beyond its citation, so record the origin and checksum of your copy.

## Lung Mask and Kvasir-SEG

Images and masks must share a basename. Mask names ending in `_mask` or `-mask` are also
recognized.

```text
data/lung_mask/
  images/
    case001.png
  masks/
    case001_mask.png

data/kvasir_seg/
  images/
    cju0qkwl35piu0993l0dewei2.jpg
  masks/
    cju0qkwl35piu0993l0dewei2.jpg
```

## BraTS

The adapter discovers complete four-modality cases recursively:

```text
data/brats/BraTS20_Training_001/
  BraTS20_Training_001_flair.nii.gz
  BraTS20_Training_001_t1.nii.gz
  BraTS20_Training_001_t1ce.nii.gz
  BraTS20_Training_001_t2.nii.gz
  BraTS20_Training_001_seg.nii.gz
```

It uses the middle axial slice by default, scales each modality by its nonzero 1st/99th
percentiles, and maps label 4 to class 3. Splitting is case-level because one item is one
case. A different slice policy is a protocol change and should be recorded in the config.

## Licenses and access

This repository does not redistribute images. BraTS, ChestX-ray14, iNaturalist, and other
datasets may require registration, agreement to terms, or manual download. Their licenses
override the software license in this repository.
