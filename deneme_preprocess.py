import pandas as pd
import re
import nltk
import random
from nltk.corpus import stopwords

# NLTK Stopwords ve WordNet indirilmesi
print("Downloading necessary NLTK resources...")
nltk.download('stopwords')
nltk.download('punkt')

# Yeni veri setini yükleme
print("Loading dataset...")
file_path = "emotion_dataset_raw.csv"
data = pd.read_csv(file_path)
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# 'neutral' sınıfını veri setinden çıkarma
print("Removing 'neutral' class from dataset...")
data = data[data['Emotion'] != 'neutral']
print(f"Dataset after removing 'neutral' class: {data.shape[0]} rows.")

# Sınıf dengesini sağlama
print("Balancing dataset...")
emotion_counts = data['Emotion'].value_counts()
max_count = emotion_counts.max()

balanced_data = pd.DataFrame()
for emotion, count in emotion_counts.items():
    samples = data[data['Emotion'] == emotion]
    if count < max_count:
        sampled_data = samples.sample(max_count, replace=True, random_state=42)
    else:
        sampled_data = samples
    balanced_data = pd.concat([balanced_data, sampled_data], ignore_index=True)
print(f"Balanced dataset has {balanced_data.shape[0]} rows.")

# Veri augmentasyonu fonksiyonu
def frequent_word_insertion(text, n=1, frequent_words=None):
    """Random insertion with only frequent words"""
    words = nltk.word_tokenize(text)
    for _ in range(n):
        if frequent_words:
            random_word = random.choice(frequent_words)
        else:
            random_word = random.choice(words)
        position = random.randint(0, len(words))
        words.insert(position, random_word)
    return " ".join(words)

# En sık kullanılan kelimeleri hesapla
all_words = " ".join(balanced_data['Text'].dropna()).split()
word_freq = pd.Series(all_words).value_counts()
frequent_words = word_freq[word_freq > 5].index.tolist()  # En az 5 kez geçen kelimeler

# Veri augmentasyonu uygulama
print("Applying data augmentation...")
augmented_data = []
for index, row in balanced_data.iterrows():
    original_text = row["Text"]
    emotion = row["Emotion"]
    augmented_data.append({"Text": original_text, "Emotion": emotion})
    augmented_data.append({"Text": frequent_word_insertion(original_text, frequent_words=frequent_words), "Emotion": emotion})

augmented_df = pd.DataFrame(augmented_data)
print(f"Augmented dataset has {augmented_df.shape[0]} rows before filtering.")

# Duplicate Check: Remove Exact Duplicates
augmented_df = augmented_df.drop_duplicates(subset=["Text"])
print(f"Dataset after removing duplicates: {augmented_df.shape[0]} rows.")

# Veri temizleme fonksiyonu
def clean_text(text):
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Noktalama işaretlerini kaldır
    text = text.strip()  # Boşlukları temizle
    words = nltk.word_tokenize(text)  # Tokenize et
    words = [word for word in words if word not in stopwords.words("english")]  # Stopwords kaldır
    return " ".join(words)

# Metinleri temizleme
print("Cleaning text data...")
augmented_df["Cleaned_Text"] = augmented_df["Text"].apply(clean_text)
print("Text cleaning complete.")

# Filter Short and Long Texts
min_len = 5  # Minimum kelime sayısı
max_len = 50  # Maksimum kelime sayısı
augmented_df = augmented_df[augmented_df["Cleaned_Text"].apply(lambda x: min_len <= len(x.split()) <= max_len)]
print(f"Dataset after filtering short and long texts: {augmented_df.shape[0]} rows.")

# Sınıf dağılımını kontrol etme
print("Final class distribution:")
print(augmented_df["Emotion"].value_counts())

# İşlenmiş veriyi kaydetme
output_path = "processed_emotion_dataset.csv"
augmented_df.to_csv(output_path, index=False)
print(f"Processed dataset saved to {output_path}")
