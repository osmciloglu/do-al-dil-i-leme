ÇALIŞTIRMA TALİMATLARI ; 

1. Gerekli kütüphaneleri yükle
İlk hücreye pandas, sklearn, numpy gibi kütüphaneleri içe aktaran kodları yaz. Bu kütüphaneler olmadan TF-IDF ve benzerlik hesaplaması yapılamaz.

2. investigations.csv dosyasını Jupyter Notebook ortamına yükle
Dosyan bilgisayarında varsa aynı klasöre koy ve pd.read_csv("investigations.csv") komutunu kullanarak oku. Dosya yolu doğru olmalı.

3. SUMMARY sütunundaki eksik verileri temizle
Bazı satırlarda özet metni eksik olabilir. Bu nedenle fillna("") ile boş metinlerle doldurman gerekir. Aksi halde TF-IDF kodu hata verir.

4. TF-IDF vektörleştirme işlemini yap
TfidfVectorizer kullanarak metinleri sayısal vektörlere dönüştür. Bu adımda İngilizce durdurma kelimeleri hariç tutulur ve maksimum 1000 kelime ile sınırlı tutulur.

5. Cosine similarity matrisini oluştur
cosine_similarity fonksiyonu ile her metin ile diğer metinler arasındaki benzerliği hesapla. Sonuç, metin sayısı kadar satır ve sütun içeren bir benzerlik matrisi olur.

6. İlk metin için en benzer 5 metni bul
Benzerlik matrisinden 0. sıradaki metinle en yüksek benzerliğe sahip diğer 5 metni sırala. np.argsort fonksiyonunu kullanarak en yüksek skorlara göre sıralama yap.

7. Bu benzer metinleri ve skorlarını ekranda göster
Sıraladığın ilk 5 metnin hem indeksini, hem benzerlik skorunu hem de özet metnini ekrana yazdır. Böylece hangilerinin en çok benzediğini anlamış olursun.

Gerekli Kütüphaneleri İçe Aktar
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

Veri Setini Yükle
python
df_investigations = pd.read_csv("investigations.csv")

3. Eksik Değerleri Temizle
python
df_investigations['SUMMARY'] = df_investigations['SUMMARY'].fillna("")

4. TF-IDF Vektörleştirme
python
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df_investigations['SUMMARY'])

5. Cosine Similarity Hesapla
python
similarity_matrix_tfidf = cosine_similarity(tfidf_matrix)

6. İlk Metin İçin En Benzer 5 Metni Göster
python
similarities = similarity_matrix_tfidf[0]
similar_indices = np.argsort(similarities)[::-1][1:6]  # ilk metin hariç

for idx in similar_indices:
    print(f"Metin {idx} - Benzerlik Skoru: {similarities[idx]:.4f}")
    print("Metin İçeriği:")
    print(df_investigations['SUMMARY'].iloc[idx])
    print("-" * 100)

    Bu kodları sırasıyla hücrelere yazarak ve çalıştırılabilir. Sonuç olarak, investigations.csv içindeki ilk özet metne en çok benzeyen 5 metni ve benzerlik skorlarını görebiliriz.
