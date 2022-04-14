import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage import io
from skimage.transform import resize


def get_model():
    model = tf.keras.models.load_model('saved_model_for_interpretability')
    return model


# Clases
classes = ['bathroom', 'bedroom', 'dinning', 'frontal', 'kitchen', 'livingroom']


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(image, model):
    x = np.expand_dims(image, axis=0)

    y = model.predict(x)
    return y


def get_img_array(img):
    size = (384, 384)

    resized_image = resize(img, size)
    # Convert the image to a 0-255 scale.
    rescaled_image = 255 * resized_image

    array = rescaled_image.astype(np.uint8)

    return array


def make_gradcam_heatmap(img_array, model, pred_index=None):
    last_conv_layer_name = "top_conv"
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_gradcam_img(img, heatmap, cam_path="grad_cam_result.jpg", alpha=0.4):
    # Load the original image
    # img = tf.keras.preprocessing.image.load_img(img_path)
    # img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))

    # superimposed_img.show()
    return superimposed_img


def main():

    st.set_page_config(layout="wide")

    if 'model' not in st.session_state:
        model = get_model()
        st.session_state['model'] = model
    else:
        model = st.session_state['model']

    st.title("Keras CNN Interpretbility")

    col1, col2, col3 = st.columns(3)

    with col2:
        url = st.text_input('Pon la URL de una imagen')
        if url:
            image = io.imread(url)
            st.header("Imagen Cargada")
            st.image(image, caption="Imagen", use_column_width=False)

    with col3:
        img_file_buffer = st.file_uploader("Carga una imagen")
        # El usuario carga una imagen
        if img_file_buffer is not None:
            image = io.imread(img_file_buffer)
            st.header("Imagen Cargada")
            st.image(image, use_column_width=True, width=10)

    with col1:
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predecir..."):

            preprocessed_image = get_img_array(image)
            prediction = model_prediction(preprocessed_image, model)
            classes_predictions = dict(zip(classes, prediction[0]))
            sorted_tuples = sorted(classes_predictions.items(), key=lambda item: item[1], reverse=True)
            st.info(f'Predicciones: {sorted_tuples}')
            st.success(f'Mejor Classe: {classes[np.argmax(prediction)]}')

            # Grad-CAM Algotithm

            # Remove last layer's softmax
            model.layers[-1].activation = None
            # Generate class activation heatmap
            preprocessed_image_expanded = np.expand_dims(preprocessed_image, axis=0)
            heatmap = make_gradcam_heatmap(preprocessed_image_expanded, model)

            grad = get_gradcam_img(preprocessed_image, heatmap)

            st.header("Grad-CAM Result")
            st.image(grad, use_column_width=True, width=10)

            st.header("Heatmap")
            plt.matshow(heatmap)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()


if __name__ == '__main__':
    main()
