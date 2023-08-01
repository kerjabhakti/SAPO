# SAPO
Sentimen analisis aplikasi POS aja dari Playstore menggunakan SVM (Support Vector Machine)
1. Codingan scrapping
2. Dataaset dari Pos Aja
3. Hasil Sentimen analisisnya
4. Masukan pada README.md untuk masing-masing langkahnya

# Langkah yang harus dilakukan
1. Instal google play scrapper
2. Tampilkan data
3. Preprocessing
4. Running process SVM
5. Akurasi

## Tahap pertama install googl play scrapper
Caranya begini
``` python
!pip install google-play-scraper
```
![image](https://github.com/kerjabhakti/SAPO/assets/15622730/b0aadf63-df01-4a2c-a40a-0056272a48e2)
Selanjutnya
``` python
from google_play_scraper import app

import pandas as pd

import numpy as np
     

from google_play_scraper import Sort, reviews_all

result = reviews_all(
    'com.posindonesia.cob',
    sleep_milliseconds=0, # defaults to 0
    lang='id', # defaults to 'en'
    country='id', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT , you can use Sort.NEWEST to get newst reviews
)
     

df_busu = pd.DataFrame(np.array(result),columns=['review'])

df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist()))

df_busu.head()
```
Maka hasilnya akan keluar seperti table berikut![image_2023-08-01_074337385](https://github.com/kerjabhakti/SAPO/assets/56683476/e0979c5b-66f6-4079-b23e-5013e90aa1fd)

Selanjutnya masukan
``` python
len(df_busu.index) #count the number of data we got
```
hasilnya akan keluar jumlah datanya seperti berikut:
![image_2023-08-01_074707459](https://github.com/kerjabhakti/SAPO/assets/56683476/241153b4-839d-4cd4-9dc4-47aec15765c4)

Dilanjutkan dengan penampilan table yang penting saja
``` python
df_busu[['userName', 'score','at', 'content']].head()  #preview userName, rating, date-time, and reviews only
```

![image_2023-08-01_074818868](https://github.com/kerjabhakti/SAPO/assets/56683476/a105d049-b7e8-4781-acd9-804039d0322d)
lalu di rapikan dengan
``` python
new_df = df_busu[['userName', 'score','at', 'content']]
sorted_df = new_df.sort_values(by='at', ascending=False) #Sort by Newst, change to True if you want to sort by Oldest.
sorted_df.head()!

```
[image_2023-08-01_075018443](https://github.com/kerjabhakti/SAPO/assets/56683476/53c19557-7a42-4a5e-964f-c08474b876c6)

dilanjutkan dengan cara berikut agar terlihat rapi
``` python
my_df = sorted_df[['userName', 'score','at', 'content']] #get userName, rating, date-time, and reviews only

my_df.head()
```
![image_2023-08-01_075220777](https://github.com/kerjabhakti/SAPO/assets/56683476/fb378d0e-69b3-42a3-9a9f-d257ec0ddbef)

Lalu save filenya ke CSV dengan cara berikut
``` python
my_df.to_csv("scrapped_data.csv", index = False)
```

Pilih kolom yang akan dipakai
``` python
my_df=my_df[['content', 'score']]
```
Dilajutkan dengan pelabelan
``` python
def pelabelan(score):
  if score < 3:
    return 'Negatif'
  elif score > 3 :
    return 'Positif'
  elif score == 3 :
    return 'Netral'
my_df['Label'] = my_df ['score'].apply(pelabelan)
my_df.head(100)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/0b074463-c2f6-456e-9e2b-f92af761c838)

lalu disave dengan
``` python
my_df.to_csv("scrapped_data.csv", index = False)
```

Langkah selanjutnya adalah Cleaning data
masukan
``` python
import pandas as pd
pd.set_option('display.max_columns', None)
my_df = pd.read_csv('/content/scrapped_data.csv')
my_df.head(100)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/71bdf7c1-f330-4b20-88f9-c1067485353b)

lalu untuk menampilkan info
``` python
my_df.info()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/47ce36d9-5911-4f79-bc26-9762e655d22c)

Lalu melihat baris yang nilainya null
``` python
my_df.isna()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/543fba53-b567-4f4a-be9a-cac68d83aae1)

masukan index describe ascendsing (false)
``` python
my_df.isna().any()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/cc45d01c-0ecf-4330-a30a-89b7d4ebe032)

Dilanjutkan dengan mendeskripsikan
``` python
my_df.describe()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/493c0e77-95a8-43e1-a3e5-93948bda9906)

Dilanjutkan dengan mencari baris yang nilainya null
``` python
my_df.isnull().sum()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/fae2d402-3bab-4917-93b1-c6351b4008bf)

Langkah selanjutnya adalah Handling missing value-ignore tuple
masukkan codingan
``` python
my_df.dropna(subset=['Label'],inplace = True)

my_df.isnull().sum()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/44f2c220-0f05-4c11-a8fa-7b1a43fa8afe)

Tampilakan 50 data teratas
``` python
my_df.head(50)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/fd599b83-c99c-4461-b65a-0ea82c0b4906)

lalu simpan ke dalam csv
``` python
my_df.to_csv("posajapreprocessing.csv", index = False)  
```

Langkah selanjutnya adalah Text PreProcessing
``` python
import pandas as pd
df = pd.read_csv('/content/posajapreprocessing.csv')
df.head(50)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/28b86708-3893-43f4-9e55-e2054a230324)

Dilanjutkan dengan CaseFolding
``` python
import re
def  clean_text(df, text_field, new_text_field_name):
    my_df[new_text_field_name] = my_df[text_field].str.lower()
    my_df[new_text_field_name] = my_df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # remove numbers
    my_df[new_text_field_name] = my_df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return my_df
my_df['text_clean'] = my_df['content'].str.lower()
my_df['text_clean']
data_clean = clean_text(my_df, 'content', 'text_clean')
data_clean.head(10)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/200d6389-e9c5-4517-a5ac-94084593c2d5)

Lalu Dilanjutkan dengan Stopword Removal
``` python
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('indonesian')
data_clean['text_StopWord'] = data_clean['text_clean'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
data_clean.head(100)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/b7b87e24-37ac-4182-8c73-addcfec229b4)

Lalu membuat WordCloud
``` python
!pip install wordcloud
data_komentar = ' '.join(data_clean['text_StopWord'])
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Membuat word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(data_komentar)

# Menampilkan word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/0ad5101d-b3cd-46e0-94a3-374f024191ef)

Langkah Selanjutnya Tokenaizing
``` python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
data_clean['text_tokens'] = data_clean['text_StopWord'].apply(lambda x: word_tokenize(x))
data_clean.head()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/b9476939-69ec-4f07-acb1-38858e5eed5f)
 DIlanjutkan dengan Stemming
``` python
!pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
#-----------------STEMMING -----------------
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}
hitung=0

for document in data_clean['text_tokens']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '

print(len(term_dict))
print("------------------------")
for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    hitung+=1
    print(hitung,":",term,":" ,term_dict[term])

print(term_dict)
print("------------------------")

# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]


#script ini bisa dipisah dari eksekusinya setelah pembacaaan term selesai
data_clean['text_steamindo'] = data_clean['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x)))
data_clean.head(20)
```
Lalu outputnya kurang lebih seperti ini
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/17084c65-5ccc-4728-80c6-e45328147d4f)
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/a333fb2e-421f-4022-9a57-7887a120afcd)

lalu simpan ke Csv file
``` python
data_clean.to_csv('hasil_TextPreProcessing_posaja.csv', index= False)
```
Mengimport Library reuntuk prapose
``` python
import re
def praproses(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)()(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text
```
lalu splitting data
``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_clean['content'], data_clean['Label'],
                                                    test_size = 0.20,
                                                    random_state = 0)
```

Dilanjutkan dengan pembobotan tf-idf
``` python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/48edbaaf-8cba-46ef-aff9-94c00baa323f)

lalu lanjutkan dengan menghitung vektorasi
``` python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X_train)
```
``` python
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

X_train.toarray()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/ccf9489e-4a67-449d-83d8-c9444804c356)

Barulah kita masuk ke metode SVM
``` python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Membaca data komentar dan sentimen dari file CSV
data = pd.read_csv('hasil_TextPreProcessing_posaja.csv')

# Memisahkan teks komentar dan sentimen
X = data['content']
y = data['Label']

# Pra-pemrosesan data menggunakan CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Memisahkan data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Memprediksi sentimen komentar pada data pengujian
y_pred = svm_model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
```
Mendapatkan hasil akurasi 
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/a0feda18-c638-45c8-9c88-1ffab5f80aed)

Tampilkan data kedalam bentuk grafik
``` python
import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file CSV
df = pd.read_csv('hasil_TextPreProcessing_posaja.csv')

# Menghitung jumlah sentimen
sentimen_count = df['Label'].value_counts()

# Menampilkan grafik batang
plt.bar(sentimen_count.index, sentimen_count.values)
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.title('Grafik Sentimen Komentar')
plt.show()
```
![image](https://github.com/kerjabhakti/SAPO/assets/56683476/7b2ed6d6-9c1f-48c5-82f4-04d274985919)



