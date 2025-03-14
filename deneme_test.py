import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Parametreler
model_path = "model_checkpoints/emotion_model_deep.h5"
tokenizer_path = "datasets/tokenizer2.pkl"
label_encoder_path = "datasets/label_encoder.pkl"
max_len = 100

# Model, tokenizer ve label encoder'i yukleme
print("Loading model, tokenizer, and label encoder...")
model = load_model(model_path)
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
with open(label_encoder_path, 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)
print("Model, tokenizer, and label encoder loaded successfully.")

# Kullanici girdisini isle ve tahmin yap
def predict_emotion(text):
    # Metni isleme
    user_input_seq = tokenizer.texts_to_sequences([text])
    user_input_padded = pad_sequences(user_input_seq, maxlen=max_len, padding="post")

    # Tahmin yapma
    prediction = model.predict(user_input_padded)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

if __name__ == "__main__":
    print("Emotion Prediction System")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("Enter a sentence: ")
        if user_input.lower() == 'exit':
            print("Exiting... Goodbye!")
            break
        
        predicted_emotion = predict_emotion(user_input)
        print(f"Predicted Emotion: {predicted_emotion}")
