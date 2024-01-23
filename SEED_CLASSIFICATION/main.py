import streamlit as st 
import tensorflow as tf 
import numpy as np 
import time


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single Image to batch
    predictions = model.predict(input_arr)

    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    accuracy_percent = confidence * 100
    return predicted_class_index, accuracy_percent

#Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page',['Home','About Project','Prediction'])

#Home Page
if(app_mode=='Home'):
    st.header('SEEDS CLASSIFICATION SYSTEM')
    image_path = 'bg_frvg.jpg'
    st.image(image_path)

#About Our Project Page
elif(app_mode=='About Project'):
    st.header('About Our Project')
    st.text('We Predict Seeds Images')
    st.subheader('About Dataset')
    st.code('A data set is a collection of related, discrete items of related data that may be accessed individually or in combination or managed as a whole entity.')
    st.text('This dataset contains images of the following food items:')
    st.code('fruits Seed- melon, singkamas, sweet corn, watermelon, waxyc corn.')
    st.code('vegetables Seed- kalabasa, pechay, red pole sitao, snap beans, snow peas.')
    st.subheader('Content')
    st.text('This dataset contains three folders:')
    st.code('1. train (500 images each)')
    st.code('2. test (50 images each)')
    st.code('3. validation (50 images each)')

# Prediction Page
elif app_mode == 'Prediction':
    st.header("Let's try to Predict")

    # Option to upload a custom image
    test_image = st.file_uploader('Or choose from the predefined images:', type=['jpg', 'jpeg', 'png'])

    # Initialize selected_predefined_image
    selected_predefined_image = "seed_ex_1.jpg"

    if test_image is not None:
        # Check if the uploaded image is a seed image
        seed_image_types = ['jpg', 'jpeg', 'png']
        file_extension = test_image.name.split('.')[-1].lower()

        if file_extension in seed_image_types:
            if st.button('Show Image', key='show_image_button'):
                st.image(test_image, width=4, use_column_width=True)
                # Update selected_predefined_image when a new image is uploaded
                selected_predefined_image = None  # Set to None to prevent showing predefined image simultaneously

    # Or choose from predefined images
    predefined_images = [
        "seed_ex_1.jpg",
        "seed_ex_2.jpg",
        "seed_ex_3.jpg",
    ]

    # Handle the case when selected_predefined_image is None
    if selected_predefined_image is None:
        selected_predefined_image = st.selectbox('Select an example image:', predefined_images)
    else:
        selected_predefined_image = st.selectbox('Select an example image:', predefined_images, index=predefined_images.index(selected_predefined_image))

    # Show the selected predefined image
    if st.button('Show Example Image'):
        st.image(selected_predefined_image, width=4, use_column_width=True)

    # Predict Button
    if st.button('Predict'):
        with st.spinner("Predicting..."):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)  # Simulate processing time
                progress_bar.progress(percent_complete + 1)

        if test_image is not None:
            result_index, accuracy_percent = model_prediction(test_image)  # Use the uploaded image for prediction
        else:
            result_index, accuracy_percent = model_prediction(selected_predefined_image)  # Use selected_predefined_image for prediction

        # Reading Labels
        with open("labels (1).txt") as f:
            content = f.readlines()
        label = [i[:-1] for i in content]
        st.success("Model is predicting it's a/an {} with {:.2f}% accuracy.".format(label[result_index], accuracy_percent))
    else:
        st.warning("Note: Please upload a valid seed image (JPEG or PNG format).")
        # Add a note about uploading clear and close-up images
        st.info("Note: Please upload a clear and close-up image of a seed for accurate classification.")
