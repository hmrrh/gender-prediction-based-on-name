import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def tokenize_name(name, tokenizer, max_len):
    name_sequence = tokenizer.texts_to_sequences([name.lower().split()])
    padded_sequence = pad_sequences(name_sequence, maxlen=max_len, padding='post')
    return padded_sequence

loaded_model = load_model('model/model_modifikasi.h5')
with open('model/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer, loaded_max_len = pickle.load(handle)

nama_uji = "Dewa Sheva Dzaky"
tokenized_name = tokenize_name(nama_uji, loaded_tokenizer, loaded_max_len)
predictions = loaded_model.predict(tokenized_name)
predicted_gender = 1 if predictions[0] > 0.5 else 0
confidence_score = predictions[0][0] if predicted_gender else 1 - predictions[0][0]

print(f"Nama: {nama_uji}")
print(f"Prediksi Gender: {'Laki-Laki' if predicted_gender else 'Perempuan'}")
print(f"Confidence Score: {confidence_score}")
