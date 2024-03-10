import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

def tokenize_name(name, tokenizer, max_len):
    name_sequence = tokenizer.texts_to_sequences([name.lower().split()])
    padded_sequence = pad_sequences(name_sequence, maxlen=max_len, padding='post')
    return padded_sequence

def load_and_predict(model_path, name, token):
    loaded_model = load_model(model_path)
    
    with open(token, 'rb') as handle:
        loaded_tokenizer, loaded_max_len = pickle.load(handle)

    tokenized_name = tokenize_name(name, loaded_tokenizer, loaded_max_len)
    predictions = loaded_model.predict(tokenized_name)
    predicted_gender = 1 if predictions[0] > 0.5 else 0
    confidence_score = predictions[0][0] if predicted_gender else 1 - predictions[0][0]
    return predicted_gender, confidence_score

def main():
    st.title("Gender Prediction App")
    st.sidebar.title("Menu")
    selected_page = st.sidebar.selectbox("Choose a page", ["Home", "Predict", "Training Results"])

    if selected_page == "Home":
        st.header("Welcome to the Gender Prediction App!")
        st.write("This app predicts the gender associated with a given name. It is developed as a part of the Natural Language Processing (NLP) course assignment, and the images below showcase the members of our group.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image("image/profile/1.jpg", caption="M. Agil Faturrahman", use_column_width=True)

        with col2:
            st.image("image/profile/2.jpg", caption="Bariq Khairullah ", use_column_width=True)

        with col3:
            st.image("image/profile/3.jpg", caption="Indra Juliansyah Putra", use_column_width=True)

        with col4:
            st.image("image/profile/4.jpg", caption="Ramadhania Humaira", use_column_width=True)

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.image("image/profile/5.jpg", caption="Aditya Kurniawan", use_column_width=True)

        with col6:
            st.image("image/profile/6.jpg", caption="Affandi Arrizal", use_column_width=True)

        with col7:
            st.image("image/profile/7.jpg", caption="Moh. Surya Ejato", use_column_width=True)
        
        st.write("Explore the app by selecting different pages from the sidebar menu:")
        
        st.markdown("- **Home:** View this welcome message.")
        st.markdown("- **Predict:** Enter a name for gender prediction.")
        st.markdown("- **Training Results:** Access and analyze training results.")

    elif selected_page == "Predict":

        name_to_predict = st.text_input("Enter a name for gender prediction:")

        if st.button("Predict Gender"):
            if name_to_predict:
                predicted_gender, confidence_score = load_and_predict("model/model_modifikasi.h5", name_to_predict, 'model/tokenizer.pickle')

                st.write(f"Name: {name_to_predict}")
                st.write(f"Predicted Gender: {'Laki-Laki' if predicted_gender else 'Perempuan'}")
                st.write(f"Confidence Score: {confidence_score:.2f}")

    else:
        st.header("Training Results")
        st.image("image/accuracy.png")
        st.image("image/loss.png")

if __name__ == "__main__":
    main()
