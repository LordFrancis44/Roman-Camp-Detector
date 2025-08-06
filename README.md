# Roman-Camp-Detector
# A Multi-Spectral Approach for Roman Camp Detection

## Project Overview

This project presents an end-to-end machine learning pipeline for the detection of potential Roman military marching camps using multi-spectral satellite imagery. Leveraging free, high-resolution data from the European Space Agency's Copernicus Programme, this work demonstrates a complete workflow from data acquisition and processing to model training and deployment of an interactive detection application.

The methodology is centered on a modified ResNet18 convolutional neural network, adapted to process 4-channel spectral data (Blue, Green, Red, and Near-Infrared). This approach moves beyond simple color analysis to identify subtle vegetation anomalies—known as cropmarks—which are often indicative of underlying archaeological features. The final output is a prototype web application, built with Streamlit, that scans a target satellite image and highlights areas with a high probability of containing undiscovered archaeological sites.

This repository serves as a portfolio piece showcasing skills in geospatial data analysis, computer vision, deep learning with PyTorch, and ML application deployment.

## Table of Contents
- [Methodology](#methodology)
  - [1. Data Acquisition and Pre-processing](#1-data-acquisition-and-pre-processing)
  - [2. Dataset Creation](#2-dataset-creation)
  - [3. Model Architecture and Training](#3-model-architecture-and-training)
  - [4. Detection and Deployment](#4-detection-and-deployment)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Execution](#execution)
- [Future Work and Potential Improvements](#future-work-and-potential-improvements)

## Methodology

The project is structured into four distinct phases, constituting a complete ML pipeline.

### 1. Data Acquisition and Pre-processing

- **Satellite Imagery:** Data was sourced from the **Sentinel-2 mission**, accessed via the **Copernicus Data Space Ecosystem**. The target area focuses on the region surrounding Hadrian's Wall in Northern England, a locus of significant Roman military activity. Level-2A products were chosen, as they provide atmospherically corrected surface reflectance data.
- **Spectral Band Selection:** To enable advanced vegetation analysis, four spectral bands at 10-meter resolution were utilized:
    - **B02:** Blue (490 nm)
    - **B03:** Green (560 nm)
    - **B04:** Red (665 nm)
    - **B08:** Near-Infrared (NIR) (842 nm)
- **Data Stacking:** The individual band files (`.jp2`) were loaded using the `rasterio` library and stacked into a single 4-channel GeoTIFF (`.tif`) file. This unified data source served as the basis for all subsequent analysis.

### 2. Dataset Creation

- **Ground Truth Data:** A foundational dataset of known Roman camp locations was compiled from publicly available archaeological records, such as Historic England's National Heritage List. The coordinates for these sites were transformed from their native coordinate reference system (e.g., British National Grid) to the satellite image's UTM projection (EPSG:32630) using `pyproj`.
- **Sample Extraction:**
    - **Positive Samples (`camps`):** For each known camp location, a 500x500 meter square image tile (50x50 pixels) was extracted from the 4-channel GeoTIFF. These represent the features our model must learn to identify.
    - **Negative Samples (`no_camps`):** A set of random 500x500 meter tiles was generated from areas at a safe distance (>1 km) from any known site. These samples—representing generic farmland, forests, and other terrain—teach the model what to ignore.
- **Data Storage:** The extracted multi-channel tiles were saved as NumPy arrays (`.npy`), an efficient format for storing numerical data.

### 3. Model Architecture and Training

- **Framework:** The deep learning model was built and trained using **PyTorch**.
- **Custom Data Loader:** A custom `Dataset` class was implemented to handle the loading of `.npy` files, as the standard `ImageFolder` class is limited to image formats like PNG/JPEG.
- **Model Modification:** A **ResNet18** architecture, pre-trained on the ImageNet dataset, was chosen as the backbone. Two critical modifications were made for this specific task:
    1.  **Input Layer Adaptation:** The first convolutional layer (`conv1`) was replaced with a new `Conv2d` layer capable of accepting **4 input channels** instead of the standard 3. The pre-trained weights for the R, G, and B channels were preserved, while the weights for the new NIR channel were initialized as the mean of the other three, providing a sensible starting point for a process known as *transfer learning*.
    2.  **Output Layer Adaptation:** The final fully-connected layer (`fc`) was replaced with a new linear layer with only **2 outputs**, corresponding to our two classes: `camps` and `no_camps`.
- **Training:** The model was trained on an Apple M1 Max device, leveraging the **Metal Performance Shaders (MPS)** backend for GPU acceleration. The entire network was fine-tuned using the Adam optimizer and Cross-Entropy Loss function. The model that achieved the highest accuracy on the validation set was saved for deployment.

### 4. Detection and Deployment

- **Detection Algorithm:** A sliding window algorithm was implemented to scan the entire 4-channel GeoTIFF. The model analyzes each 500x500m tile and classifies it as either `camps` or `no_camps`.
- **Interactive Application:** A user-friendly web application was developed using **Streamlit**. The app allows a user to:
    1.  View the satellite image of the study area.
    2.  Initiate the detection process with a single button click.
    3.  Monitor the scan via a real-time progress bar.
    4.  Visualize the results as bounding boxes overlaid on the map, indicating the locations of potential sites.

## Results

*(This section is for you to fill in. It's crucial!)*

The model achieved a peak validation accuracy of **[Your Validation Accuracy Here, e.g., 92.5%]** after 15 epochs of training. 

The deployed Streamlit application successfully identifies several known camp locations and highlights new areas of interest that warrant further investigation. The following image shows the output of the detection process:

![Detection Results](path/to/your/screenshot.png) 
*(**Action Required:** Replace this with a link to the screenshot of your Streamlit app's results.)*

## Project Structure
