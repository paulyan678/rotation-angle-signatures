# Dataset setup

I exclude dataset files from Git. Set `data_root` in the YAML configuration or use my
default `data/` directory. In the classification experiment I use images only; I do not
use the original class labels in the contrastive or binary-pair task.

I use `data_revision` as a stable identifier for the exact local dataset release and
manifests. Change it whenever files, subsets, or preprocessing inputs change; I include it
in every job ID to prevent stale-result reuse. For manually licensed datasets, record the
official release identifier in your local research manifest.

## Dataset versions and sources

I use the following releases. The automatic adapters combine the splits listed here so the
image counts match the study design; the classification experiment ignores class labels.

| Config key | Release and included images | Authoritative source / access |
|---|---|---|
| `cifar10` | Original CIFAR-10, train + test (60,000); archive MD5 `c58f30108f718f92721af3b95e74349a` | [University of Toronto CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html); automatic |
| `mnist` | Original MNIST IDX, train + test (70,000) | [Yann LeCun's MNIST page](https://yann.lecun.com/exdb/mnist/); automatic |
| `fashion_mnist` | Zalando Fashion-MNIST, train + test (70,000) | [Official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist); automatic |
| `tiny_imagenet` | Tiny ImageNet-200 (2015), training images only (100,000) | [Stanford CS231n archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip); manual |
| `brats` | BraTS 2020 Training, four MRI modalities + segmentation labels | [CBICA BraTS 2020 data request](https://www.med.upenn.edu/cbica/brats2020/data.html); registration required |
| `lung_mask` | The paired chest X-ray/mask collection used in the study (approximately 1,000 pairs) | The workshop citation does not expose a stable archive ID; place the same licensed collection in the layout below and record its checksum in `data_revision` |
| `kvasir_seg` | Kvasir-SEG 2020, 1,000 image/mask pairs | [Official Simula Kvasir-SEG download](https://datasets.simula.no/kvasir-seg/); manual |
| `stanford_dogs` | Stanford Dogs / ImageNet Dogs, 20,580 images | [Stanford Vision dataset page](http://vision.stanford.edu/aditya86/ImageNetDogs/); manual |
| `inaturalist` | iNaturalist 2018 Competition train + validation images | [Official iNat 2018 competition repository](https://github.com/visipedia/inat_comp/blob/master/2018/README.md); manual, non-commercial research terms |
| `plant_village` | Original PlantVillage color corpus, 54,306 images | [Official PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset); manual, pin a commit |
| `chest_xray14` | NIH ChestX-ray14 expanded 2017 release, 112,120 PNGs | [NIH ChestX-ray14 archive](https://nihcc.app.box.com/v/ChestXray-NIHCC); manual |
| `svhn` | Cropped Digits train + test, excluding `extra` (99,289) | [Stanford SVHN](http://ufldl.stanford.edu/housenumbers/); automatic |
| `eurosat` | EuroSAT RGB v2, 27,000 images; archive MD5 `f46e308c4d50d4bf32fedad2d3d62f3b` | [Zenodo record 7711810](https://zenodo.org/records/7711810); automatic |
| `caltech101` | Caltech-101 v1.0 including `BACKGROUND_Google` (9,146) | [CaltechDATA record](https://data.caltech.edu/records/mzrjq-6wc02); manual |
| `flowers102` | Oxford Flowers-102, official train + validation + test (8,189) | [Oxford VGG Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/); automatic |
| `food101` | ETH Food-101, official train + test (101,000) | [ETH Zürich Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/); automatic |

I resolve the BraTS naming ambiguity in the workshop paper in favor of BraTS 2020 because
Figure 4 identifies BraTS2020 and the project uses the `BraTS20_...` case convention. The
Lung Mask citation remains the one dataset whose public archive cannot be uniquely inferred
from the paper; I keep that source explicit rather than silently substituting a different
Kaggle release.

## Automatically downloaded by torchvision

My adapters request official torchvision downloads for:

- CIFAR-10
- MNIST
- Fashion-MNIST
- SVHN
- EuroSAT
- Flowers-102
- Food-101

I require iNaturalist 2018 to be downloaded manually because its size and terms make
silent download inappropriate. I scan its images recursively because this experiment does
not use class labels.

## Image-folder datasets

I scan the following classification datasets recursively for PNG, JPEG, TIFF, or BMP
images. Put files directly below the named root or in an `images/` folder.

```text
data/
  tiny_imagenet/
  stanford_dogs/
  plant_village/
  chest_xray14/
  inaturalist/
  caltech101/
```

For Tiny ImageNet, keep only the official `train/` directory under
`data/tiny_imagenet/train/`; my adapter deliberately excludes validation and test images to
match the 100,000-example study row. For PlantVillage, put only `raw/color/` below
`data/plant_village/images/`. For Caltech-101, put the full `101_ObjectCategories/`
contents, including `BACKGROUND_Google`, below `data/caltech101/images/`.

To reproduce my experiments, acquire the dataset versions cited in the paper and preserve
their official splits outside this repository. Record the origin and checksum of each local
copy in your research manifest.

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

My adapter discovers complete four-modality cases recursively:

```text
data/brats/BraTS20_Training_001/
  BraTS20_Training_001_flair.nii.gz
  BraTS20_Training_001_t1.nii.gz
  BraTS20_Training_001_t1ce.nii.gz
  BraTS20_Training_001_t2.nii.gz
  BraTS20_Training_001_seg.nii.gz
```

I use the middle axial slice by default, scale each modality by its nonzero 1st/99th
percentiles, and map label 4 to class 3. I split at case level because one item is one case.
Record a different slice policy as a protocol change in the config.

## Licenses and access

I do not redistribute images in this repository. BraTS, ChestX-ray14, iNaturalist, and
other datasets may require registration, agreement to terms, or manual download. Their
licenses override the software license I apply to my code.
