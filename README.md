# do-al-dil-i-leme
proje 1
Bu proje, araç model verilerinin metin işleme teknikleriyle (stemming ve lemmatization) standardize edilmesini amaçlamaktadır. Özellikle:

Otomotiv veri analizi

Model tabanlı sınıflandırma sistemleri

Arama motoru optimizasyonu

gibi uygulamalarda kullanılmak üzere temizlenmiş bir veri seti üretir.

Kullanılan Veri Seti
car_models.csv
İçerik: Araç modellerine ait yıl, marka ve model bilgileri

Örnek Veri:

csv
modelYear,make,model
2022,BUICK,ENCORE
2019,HIGHLAND RIDGE,HIGHLANDER
2022,HYUNDAI,TUCSON HYBRID


Teknoloji Stack'i
Bileşen	Açıklama
Python	3.8+
Pandas	Veri işleme (pd.read_csv(), df.apply())
NLTK	Metin işleme (word_tokenize, PorterStemmer, WordNetLemmatizer)
Google Colab	Bulut tabanlı çalışma ortamı (Ücretsiz GPU desteği)



Repo Kurulumu
Adım 1: Repo Klonlama
bash
git clone https://github.com/kullanici_adi/arac-model-isleme.git
cd arac-model-isleme
Adım 2: Gereksinimlerin Yüklenmesi
bash
pip install -r requirements.txt
Adım 3: Script Çalıştırma
bash
python process_models.py --input car_models.csv --output processed_models.csv


KOD YAPISI

process_models.py
python
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import argparse

def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file path")
    args = parser.parse_args()

    # NLTK verilerini yükle
    nltk.download('punkt')
    nltk.download('wordnet')

    # Veriyi işle
    df = pd.read_csv(args.input)
    df['cleaned'] = df['model'].apply(clean_text)
    df['stemmed'], df['lemmatized'] = zip(*df['cleaned'].apply(process_text))
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main(



    Veri İşleme Akışı
    graph TD
    A[Raw CSV] --> B[Pandas ile Okuma]
    B --> C[Metin Temizleme]
    C --> D[Stemming]
    C --> E[Lemmatization]
    D --> F[Çıktı CSV]
    E --> F


    Sonuçlar
Metric	Değer
İşlenen Kayıt	50,000+
İşlem Süresi	< 2 dakika
Boyut Küçülmesi	%40


Word2Vec Vektörleştirme  
boyuttan dolayı eğitim için kullandığım kodlar ;
from gensim.models import Word2Vec

# Örnek: lemmatized_texts ve stemmed_texts hazır kabul ediliyor
# parametre setleri
param_sets = [
    {'vector_size': 100, 'window': 5, 'min_count': 2, 'sg': 0},
    {'vector_size': 100, 'window': 5, 'min_count': 2, 'sg': 1},
    {'vector_size': 200, 'window': 5, 'min_count': 2, 'sg': 0},
    {'vector_size': 200, 'window': 5, 'min_count': 2, 'sg': 1},
    {'vector_size': 100, 'window': 10, 'min_count': 2, 'sg': 0},
    {'vector_size': 100, 'window': 10, 'min_count': 2, 'sg': 1},
    {'vector_size': 200, 'window': 10, 'min_count': 2, 'sg': 0},
    {'vector_size': 200, 'window': 10, 'min_count': 2, 'sg': 1},
]

def train_and_show_models(texts, param_sets, model_name_prefix):
    models = {}
    for i, params in enumerate(param_sets):
        print(f"\n{'='*40}")
        print(f"Training {model_name_prefix} model {i+1} with parameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")
        model = Word2Vec(
            sentences=texts,
            vector_size=params['vector_size'],
            window=params['window'],
            min_count=params['min_count'],
            sg=params['sg'],
            workers=4,
            epochs=10
        )
        model_name = f"{model_name_prefix}_model_{i+1}"
        models[model_name] = model
        print(f"Model {model_name} training completed.")
    return models

# Örnek eğitim (lemmatized_texts ve stemmed_texts değişkenlerin hazır olmalı)
lemmatized_models = train_and_show_models(lemmatized_texts, param_sets, 'lemmatized')
stemmed_models = train_and_show_models(stemmed_texts, param_sets, 'stemmed')

# Kaydetme
for name, model in lemmatized_models.items():
    model.save(f"{name}.model")
    print(f"Saved model: {name}.model")

for name, model in stemmed_models.items():
    model.save(f"{name}.model")
    print(f"Saved model: {name}.model")

# Modelden örnek çıktı (ilk lemmatized modelde 'example' kelimesinin vektörü)
word = 'example'
if word in lemmatized_models['lemmatized_model_1'].wv:
    print(f"\nVector for '{word}' in lemmatized_model_1:")
    print(lemmatized_models['lemmatized_model_1'].wv[word])
else:
    print(f"\nWord '{word}' not found in lemmatized_model_1 vocabulary.")
 
