import streamlit as st
from keras.models import load_model
import numpy as np

def preprocess_name(name, name_length=50):
    name = name.lower()
    name = list(name)
    name = [(max(0.0, ord(char) - 96.0)) for char in name]
    name = (name + [0.0] * name_length)[:name_length]
    return np.array([name])

def load_and_predict(model_path, name):
    model = load_model(model_path)
    preprocessed_name = preprocess_name(name)
    processed_name = np.array(preprocessed_name).reshape(1, -1, 1)
    prediction = model.predict(processed_name)
    confidence_score = prediction[0][0]

    gender = "Laki-laki" if confidence_score > 0.5 else "Perempuan"

    if gender == 'Perempuan':
        confidence_score = 1 - confidence_score

    return gender, confidence_score

def main():
    st.title('Gender Prediction Based on Name')
    st.write("'Give it a shot! Let us know, who's your name?'")
    user_input = st.text_input('Enter a name: ')
    
    if user_input:
        predicted_gender, confidence_score = load_and_predict("gender_prediction.h5", user_input)
        

        st.write(f"Name: {user_input}")
        st.write(f"Your gender is: {predicted_gender}")
        st.write(f"Accuracy: {confidence_score:.2f}")

if __name__ == "__main__":
    main()