import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
import pickle

# Preprocessed veri setini yükleme
print("Loading preprocessed dataset...")
file_path = "processed_emotion_dataset.csv"
data = pd.read_csv(file_path)
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Eğitim ve test verilerine ayırma
print("Splitting data into training and testing sets...")
X = data["Cleaned_Text"].fillna("")  # Boş değerleri boş string ile dolduruyoruz
y = data["Emotion"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} rows.")
print(f"Testing set size: {X_test.shape[0]} rows.")

# Tokenizer ve metinleri dizileştirme
print("Tokenizing text data...")
max_words = 20000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding="post")
print("Text tokenization complete.")

# Tokenizer'i kaydetme
tokenizer_path = "datasets/tokenizer3.pkl"
with open(tokenizer_path, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
print(f"Tokenizer saved to {tokenizer_path}.")

# Label Encoding
print("Encoding labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# LabelEncoder'i kaydetme
label_encoder_path = "datasets/label_encoder.pkl"
with open(label_encoder_path, 'wb') as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)
print(f"Label encoder saved to {label_encoder_path}.")

# Model Oluşturma
print("Building the LSTM model...")
with tf.device('/GPU:0'):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))  # İlk Bidirectional LSTM katmanı
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))  # İkinci Bidirectional LSTM katmanı
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Özeti
model.summary()

# Modeli Eğitme
print("Training the model on GPU...")
with tf.device('/GPU:0'):
    history = model.fit(X_train_padded, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_padded, y_test_encoded))
print("Model training complete.")

# Modeli Kaydetme
model_path = "model_checkpoints/emotion_model_last.h5"
model.save(model_path)
print(f"Trained model saved to {model_path}.")

# Model Performansı Görselleştirme
print("Visualizing training performance...")
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Modelin Performansını Değerlendirme
print("Evaluating the model...")
with tf.device('/GPU:0'):
    loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix ve Classification Report
print("Generating confusion matrix...")
y_pred_encoded = model.predict(X_test_padded)
y_pred = label_encoder.inverse_transform(y_pred_encoded.argmax(axis=1))
cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Precision ve Recall Hesaplama
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Weighted Precision: {precision:.2f}")
print(f"Weighted Recall: {recall:.2f}")

