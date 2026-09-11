# AgroSat

**Wetland–Dryland change detection from satellite imagery using deep learning.**

## Problem Statement

Manually tracking how much land in a given area has shifted between wetland and dryland over time is slow and error-prone. AgroSat is an attempt to automate this: given satellite images of the same region taken at different times, the system segments wet land from dry land in each image and estimates the area (and approximate location) of the change between them.

## Project in a Nutshell

- Uses a deep learning image segmentation model to identify wet land in a satellite image and produce a binary mask (wet vs. not-wet) for each image.
- Compares the masks of two images of the same area (e.g. taken years apart) to estimate the change in wetland area, and flags an alert when the estimated change crosses a threshold.
- Built with **TensorFlow / Keras**, **NumPy**, **scikit-image**, and **matplotlib**; images are read with **tifffile** since the source imagery is in `.tif` format.
- Developed and run as a Google Colab notebook, with the training/test dataset stored on Google Drive.

## How It Works

The core pipeline lives in [`AgroSat_Segmentation_TF.ipynb`](AgroSat_Segmentation_TF.ipynb):

1. **Data loading** – Satellite images and their corresponding ground-truth masks (`.tif` files, read with `tifffile`) are loaded from a Google Drive folder structured as `dataset/{train,test}/{image,mask}/`.
2. **Preprocessing** – Images are converted to arrays and resized to `128 x 128 x 3`; pixel values are normalized (divided by 255) inside the model via a `Lambda` layer.
3. **Model – U-Net style encoder/decoder** – A convolutional network is built with:
   - An encoder of 4 blocks (`Conv2D` → `Conv2D` → `MaxPooling2D`), with filter sizes doubling at each stage (8 → 16 → 32 → 64), followed by a bottleneck of 128 filters.
   - A decoder of 4 blocks (`Conv2DTranspose` for upsampling) that concatenates each upsampled feature map with the matching encoder feature map via skip connections.
   - A final `1x1 Conv2D` with a `sigmoid` activation producing a single-channel binary segmentation mask.
   - Compiled with the `adam` optimizer and `binary_crossentropy` loss.
4. **Training** – The model is trained with `model.fit`, using a `ModelCheckpoint` callback to save the best-performing weights (`model-tgs-salt-1.h5`), a `validation_split` of 0.1, batch size 8, and 30 epochs.
5. **Inference** – The saved model is reloaded and used to predict masks for two images of the same area. Predictions are thresholded at 0.8 to produce a clean binary (0/255) mask.
6. **Change detection** – The two binary masks are subtracted pixel-by-pixel to find where wet land has appeared or disappeared. For flagged pixels, an approximate latitude/longitude is derived from a fixed reference coordinate and a per-pixel distance conversion factor, giving a rough geographic location for the change.
7. **Area estimation** – The number of "wet" pixels in each mask is converted into a real-world area estimate using the ratio of wet pixels to total pixels (`totalPxArea`) scaled to the total area the image represents (`totalArea`). The difference between the two estimated areas gives the change in wetland area, and an alert is printed if the cumulative change exceeds a set threshold.

### Model Architecture

```python
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import *
inputs = Input((128, 128, 3))
.
.
.
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Model Summary

```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_6 (InputLayer)            [(None, 128, 128, 3) 0                                            
__________________________________________________________________________________________________
lambda_5 (Lambda)               (None, 128, 128, 3)  0           input_6[0][0]                    
__________________________________________________________________________________________________
conv2d_68 (Conv2D)              (None, 128, 128, 8)  224         lambda_5[0][0]                   
__________________________________________________________________________________________________
conv2d_69 (Conv2D)              (None, 128, 128, 8)  584         conv2d_68[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_20 (MaxPooling2D) (None, 64, 64, 8)    0           conv2d_69[0][0]                  
__________________________________________________________________________________________________
.
.
.
conv2d_84 (Conv2D)              (None, 128, 128, 8)  1160        concatenate_14[0][0]             
__________________________________________________________________________________________________
conv2d_85 (Conv2D)              (None, 128, 128, 8)  584         conv2d_84[0][0]                  
__________________________________________________________________________________________________
conv2d_86 (Conv2D)              (None, 128, 128, 1)  9           conv2d_85[0][0]                  
==================================================================================================
Total params: 485,817
Trainable params: 485,817
Non-trainable params: 0
__________________________________________________________________________________________________
```

### Change-in-Area Calculation

```python
area_of_first_image = (Num_of_WhitePixel_in_First_image/totalPxArea)*totalArea
area_of_second_image = (Num_of_WhitePixel_in_Second_image/totalPxArea)*totalArea
difference_of_area = areaPro1 - areaPro2
```

## Project Snapshot

<img src="Arch.PNG" height="300px">

- Satellite images and their corresponding masks are used as training data.
- The trained model predicts a wet/dry mask for a new satellite image, which is then used to estimate area and detect change.

## Repository Structure

| File | Description |
|---|---|
| [`AgroSat_Segmentation_TF.ipynb`](AgroSat_Segmentation_TF.ipynb) | Main notebook: data loading, U-Net model, training, inference, and change-detection logic. |
| `Arch.PNG` | Architecture / workflow diagram shown above. |
| `LICENSE` | MIT License. |
| `README.md` | This file. |

## Getting Started

The notebook is written for **Google Colab** and expects the dataset to be available on Google Drive under a path such as `dataset/train/{image,mask}` and `dataset/test/{image,mask}`, with images and masks as `.tif` files.

1. Open `AgroSat_Segmentation_TF.ipynb` in Google Colab.
2. Mount your Google Drive and point `base_dir` to your dataset location.
3. Run the notebook cells in order to install dependencies (`tifffile`), load the data, build and train the U-Net model, and run inference/change-detection on a pair of images.

### Dependencies

- TensorFlow / Keras
- tifffile
- NumPy
- scikit-image
- matplotlib

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
