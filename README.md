Vineyard detection using high resolution images (25cm/px)
===============================================================

UNDER DEVELOPMENT

# Objective

Detect vineyard land usage using convolutional network based on aerial images dataset.
The objective is to train a binary classifier that receives as input patches of the images labeled as 0-no vineyard, 1-vineyard and outcomes the probability of the category 1-vineyard.
Once the model is trained, it will be applied to the raster using window sliding to obtain a mask with the probabilities of vineyard usage in the patch.
After this, the masks will be vectorized to obtain a shapefile, that will be filtered and simplified. 

**This is just a prototype.**

# Dataset

* High resolution aerial images (25cm/px) from spanish [Plan Nacional de Ortofotografía Aérea (PNOA)](https://pnoa.ign.es/), a restricted área from the region of Castilla y León is used in this project to train the classifier. 
The images for year 2020 in the area of Castilla y León can be accessed [here](http://ftp.itacyl.es/cartografia/01_Ortofotografia/2020/). 
* LPIS: Land parcel Information System, feature files containing parcels in the area of the images, these features are used to manually select the parcels with vineyeard usage to extract patches for each category (0-no vineyard 1-vineyard). 
These files can be downloaded for the area of Castilla y León from [here](http://ftp.itacyl.es/cartografia/05_SIGPAC/2020_ETRS89/Parcelario_SIGPAC_CyL_Municipios/).

## Data preparation
The dataset will contain samples with RGB 48x48px patches. Steps:
* Select a working area, in this case the Zone of Ribera del Duero, a well known wine production area that extends across the provinces of Valladolid, Burgos, Segovia and Soria. 
* Download the PNOA images for the working area.
* Manually select samples using QGIS, use the LPIS feature files of the working area. 
* For each raster file, filter the geometries contained in it, and use these geometries to cut the raster using the LIRs  (Large Interior Rectangles) of each feature. After this step, for each sample feature wi will have a PNG file with a rectangle that can be used to extract patches with window sliding  technique. LIRs will be separated in different folder for positive and negative samples.
LIR extraction algorithm is a numba implementation taken from https://github.com/lukasalexanderweber/lir.
* Use the LIRs to extract patches using a sliding window of 48x48px and store them in a folder.  
* Get 48x48px patches (12x12m), split the dataset in train and validation (0.3 split) and create a numpy array with the images to feed the traingin process. The dataset numpy array is stored as a pickle file (dataset.npy).

To run the data set preparation use the python script

``` console
vineyard-detector/vineyard/data/run.py
```
Edit the python code and set these variables:
``` python
    raster_folder = 'Locatio nof your aerial images'
    feature_file = cfg.resource('selectedParcels/selected_parcels.shp')  # features to cut out the rasters and extract lirs and patches.
    dataset_folder = "/media/gus/data/viticola/datasets/dataset_v2"  # destination directory for lirs, patches and dataset.npy
```
# Model
Train a binary classifier that receives as input patches of the images labeled as 0-no vineyard, 1-vineyard and outcomes the probability of the category 1-vineyard.
To easy prototype the pipeline an image detection state of art model is used, the base model is re-trained using transfer learning techniquest to adapt the layers weights to the vineyard dataset.  

Take LPIS as reference to select parcel geometries

formado por piezas de viñedo seleccionadas manualmente
Extraidos patches de 48x48px etiquetados en dos conjuntos 1=viñedo 2=no viñedo


Búsqueda de ejemplos en casos frontera que puedan ser
- Placas solares
- texturas lineales o rejillas (frutales, olivar) 
- Carreteras
- Huellas de actividad agraria, con marcas verticales

Entrenar tomar modelo de referencia y hacer transfer learning


Steps:
* Use feature file to extract Lirs (Largest Interior Rectangles) from tif images, and store then in a dataset/lirs folder
* Use the lirs to extract patches 
* Store the patches as a numpy array and store some infor in a dataset.json file


# Model selection

Select a state of art imagen classification model to use as starting point for our model.
the idea behing transfer learning is to reuse the filters the model has learn to detect
features.

FixEfficientNet is a technique combining two existing techniques: The FixRes from the Facebook AI Team[2] and the EfficientNet [3] first presented from the Google AI Research Team.

The idae is to take as base for the transfer learning a state of art model that uses

Let's take as base EfficientNetB4, is a small model (75MB) compare to current top modols and the number of parameters 
needed to train is significantly less than in newer models.
https://keras.io/api/applications/efficientnet/#efficientnetb4-function

## Layer representation
Currently EfficientNet

EfficientNet

https://paperswithcode.com/sota/image-classification-on-imagenet
https://keras.io/api/applications/




https://paperswithcode.com/sota/image-classification-on-imagenet


https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html