# ml-cnn-keras-Interpretability

## Based on
- [Tensorflow Retraining an Image Classifier](https://www.tensorflow.org/hub/tutorials/tf2_image_retraining)
- [Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [tf.keras.applications.efficientnet_v2.EfficientNetV2S](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2S)

## Training Data - Rooms Dataset: 
- [https://www.kaggle.com/robinreni/house-rooms-image-dataset](https://www.kaggle.com/robinreni/house-rooms-image-dataset)
- [https://github.com/emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset)

- You can download curated dataset with this images from [here](https://drive.google.com/file/d/1PzoQAvOcXDwEmNAj_rGnTXSfu8GPuUvR/view?usp=sharing) and unzip on *training_data* folder.


## Sample result of Grad-CAM algorithm
- Generated with [this notebook](notebooks/03-Interpretability-with-Grad-CAM.ipynb)  

![Grad-CAM Result](notebooks/grad_cam_result.jpg)

## Model Diagram EfficientNetV2-S
- Generated with [this notebook](notebooks/02-plot-cnn-keras-models.ipynb)

![EfficientNetV2-S](notebooks/efficientnetv2_s_model_diagram.png)