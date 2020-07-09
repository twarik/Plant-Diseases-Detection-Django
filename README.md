# Plant-Diseases-Detection-Django
**A Web based deep learning model API for plant disease detection and classification from leaves**

First, the dataset was collected from [here](https://github.com/spMohanty/PlantVillage-Datasethttps://github.com/spMohanty/PlantVillage-Dataset) containing 54,306 images of 14 different crops, then split into 80% training and 20% validation set.

We trained a ResNet50 model via transfer learning using a Google Colaboratory GPU.

We obtained a validation accuracy of 98.47%

The model is able to distinguish between 38 different classes (crop species and health status).
