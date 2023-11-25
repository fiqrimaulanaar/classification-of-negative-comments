from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from preprocessing import preprocess


st.set_option('deprecation.showPyplotGlobalUse', False)


df = pd.read_csv('data fix/data_kalimat.csv')
df.head()

labels = ['SARA', 'PORNOGRAFI', 'PENCEMARAN_NAMA_BAIK', 'RADIKALISME']


label_sosmed = ['Instagram', 'Kaskus', 'Twitter']

kaskus = df[df['source'] == 'kaskus']
instagram = df[df['source'] == 'instagram']
twitter = df[df['source'] == 'twitter']

# pornografi
pornografi_kaskus = kaskus[kaskus['PORNOGRAFI'] == 1]
pornografi_instagram = instagram[instagram['PORNOGRAFI'] == 1]
pornografi_twitter = twitter[twitter['PORNOGRAFI'] == 1]

# sara
sara_kaskus = kaskus[kaskus['SARA'] == 1]
sara_instagram = instagram[instagram['SARA'] == 1]
sara_twitter = twitter[twitter['SARA'] == 1]

# pencemaran nama baik
pencemaran_kaskus = kaskus[kaskus['PENCEMARAN_NAMA_BAIK'] == 1]
pencemaran_instagram = instagram[instagram['PENCEMARAN_NAMA_BAIK'] == 1]
pencemaran_twitter = twitter[twitter['PENCEMARAN_NAMA_BAIK'] == 1]

# radikalisme
radikalisme_kaskus = kaskus[kaskus['RADIKALISME'] == 1]
radikalisme_instagram = instagram[instagram['RADIKALISME'] == 1]
radikalisme_twitter = twitter[twitter['RADIKALISME'] == 1]

sara = df[df['SARA'] == 1]
pornografi = df[df['PORNOGRAFI'] == 1]
pencemaran = df[df['PENCEMARAN_NAMA_BAIK'] == 1]
radikalisme = df[df['RADIKALISME'] == 1]

sara0 = df[df['SARA'] == 0]
pornografi0 = df[df['PORNOGRAFI'] == 0]
pencemaran0 = df[df['PENCEMARAN_NAMA_BAIK'] == 0]
radikalisme0 = df[df['RADIKALISME'] == 0]

with open('data fix/Model_PORNOGRAFI', 'rb') as training_model:
    m_pornografi = pickle.load(training_model)

with open('data fix/Model_RADIKALISME', 'rb') as training_model:
    m_radikalisme = pickle.load(training_model)

with open('data fix/Model_SARA', 'rb') as training_model:
    m_sara = pickle.load(training_model)

with open('data fix/Model_PENCEMARAN', 'rb') as training_model:
    m_pencemaran = pickle.load(training_model)

hasil = []
prob = []

# page = st.selectbox("Pilih Menu", ["Menu Prediksi", "Visualisasi"])

menu_samping = st.sidebar.selectbox(
    'Menu : ',
    ('Prediksi', 'Visualisasi')
)

if menu_samping == 'Prediksi':

    text = st.text_input('Kalimat :')

    text = preprocess(text)

    st.write(f'Text setelah di preprocessing : {text}')

    if m_sara.predict([text])[0] == 0:
        hasil.append('No')
    else:
        hasil.append('Yes')

    if m_pornografi.predict([text])[0] == 0:
        hasil.append('No')
    else:
        hasil.append('Yes')

    if m_pencemaran.predict([text])[0] == 0:
        hasil.append('No')
    else:
        hasil.append('Yes')

    if m_radikalisme.predict([text])[0] == 0:
        hasil.append('No')
    else:
        hasil.append('Yes')

    prob.append(np.max(m_sara.predict_proba([text])[0]))
    prob.append(np.max(m_pornografi.predict_proba([text])[0]))
    prob.append(np.max(m_pencemaran.predict_proba([text])[0]))
    prob.append(np.max(m_radikalisme.predict_proba([text])[0]))

    df = pd.DataFrame({'Komentar negatif': labels, 'Hasil': hasil,
                      'Nilai confidence ': prob})
    st.write('')
    st.write('Hasil Prediksi :')
    st.write(df)


elif menu_samping == 'Visualisasi':
    # ProfileReport(df)

    menu = st.selectbox(
        'Visualisasi', ['Antar label', 'Antar sumber', 'Word cloud'])

    if menu == 'Antar label':

        # st.write('Perbandingan 4 label')
        # fig = plt.figure(figsize=(7, 3))

        # plt.bar(labels, [sara['SARA'].sum(), pornografi['PORNOGRAFI'].sum(),
        #         pencemaran['PENCEMARAN_NAMA_BAIK'].sum(),
        #         radikalisme['RADIKALISME'].sum()],
        #         width=0.4)

        # # plt.xlabel("Label")
        # plt.ylabel("Jumlah")
        # plt.title("Perbandingan 4 Label")
        # # plt.show()

        # st.pyplot()

        plt.suptitle('Perbandingan label 0 dan 1')
        labell = [0, 1]

        plt.subplot(2, 2, 1)
        y = [pornografi0.value_counts().sum(), pornografi.value_counts().sum()]

        plt.pie(y, labels=labell, autopct='%1.1f%%')
        plt.title('Pornografi')
        # plt.legend(title='Sosial Media :')
        # plt.show()

        plt.subplot(2, 2, 2)
        y = [sara0.value_counts().sum(), sara.value_counts().sum()]

        plt.pie(y, labels=labell, autopct='%1.1f%%')
        plt.title('SARA')
        # plt.legend(title='Sosial Media :')
        # plt.show()

        plt.subplot(2, 2, 3)
        y = [pencemaran0.value_counts().sum(), pencemaran.value_counts().sum()]

        plt.pie(y, labels=labell, autopct='%1.1f%%')
        plt.title('Pencemaran Nama Baik')
        # plt.legend(title='Sosial Media :')

        plt.subplot(2, 2, 4)
        y = [radikalisme0.value_counts().sum(), radikalisme.value_counts().sum()]

        plt.pie(y, labels=labell, autopct='%1.1f%%')
        plt.title('Radikalisme')

        st.pyplot()

        # bar
        label_0 = [sara0.value_counts().sum(), pornografi0.value_counts().sum(
        ), pencemaran0.value_counts().sum(), radikalisme0.value_counts().sum()]
        label_1 = [sara.value_counts().sum(), pornografi.value_counts().sum(
        ), pencemaran.value_counts().sum(), radikalisme.value_counts().sum()]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, label_0, width, label='0')
        rects2 = ax.bar(x + width/2, label_1, width, label='1')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Jumlah')
        ax.set_title('Perbandingan label 0 dan 1')
        ax.set_xticks(x, labels, rotation=15)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

    elif menu == 'Antar sumber':
        plt.suptitle('Distribusi Label Dari Setiap Sosial Media')

        plt.subplot(2, 2, 1)
        y = [pornografi_instagram.value_counts().sum(), pornografi_kaskus.value_counts(
        ).sum(), pornografi_twitter.value_counts().sum()]

        plt.pie(y, labels=label_sosmed, autopct='%1.1f%%')
        plt.title('Pornografi')
        # plt.legend(title='Sosial Media :')
        # plt.show()

        plt.subplot(2, 2, 2)
        y = [sara_instagram.value_counts().sum(), sara_kaskus.value_counts(
        ).sum(), sara_twitter.value_counts().sum()]

        plt.pie(y, labels=label_sosmed, autopct='%1.1f%%')
        plt.title('SARA')
        # plt.legend(title='Sosial Media :')
        # plt.show()

        plt.subplot(2, 2, 3)
        y = [pencemaran_instagram.value_counts().sum(), pencemaran_kaskus.value_counts(
        ).sum(), pencemaran_twitter.value_counts().sum()]

        plt.pie(y, labels=label_sosmed, autopct='%1.1f%%')
        plt.title('Pencemaran Nama Baik')
        # plt.legend(title='Sosial Media :')

        plt.subplot(2, 2, 4)
        y = [radikalisme_instagram.value_counts().sum(), radikalisme_kaskus.value_counts(
        ).sum(), radikalisme_twitter.value_counts().sum()]

        plt.pie(y, labels=label_sosmed, autopct='%1.1f%%')
        plt.title('Radikalisme')
        # plt.legend(title='Sosial Media :')

        st.pyplot()

    elif menu == 'Word cloud':

        plt.subplot(2, 2, 1)

        text = " ".join(text.split()[1] for text in sara.text)

        word_cloud = WordCloud(
            collocations=False, background_color='white').generate(text)

        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('SARA')
        # plt.show()

        plt.subplot(2, 2, 2)

        text = " ".join(text.split()[1] for text in pornografi.text)

        word_cloud = WordCloud(
            collocations=False, background_color='white').generate(text)

        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('PORNOGRAFI')
        # plt.show()

        plt.subplot(2, 2, 3)

        text = " ".join(text.split()[1] for text in radikalisme.text)

        word_cloud = WordCloud(
            collocations=False, background_color='white').generate(text)

        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('RADIKALISME')
        # plt.show()

        plt.subplot(2, 2, 4)

        text = " ".join(text.split()[1] for text in pencemaran.text)

        word_cloud = WordCloud(
            collocations=False, background_color='white').generate(text)

        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('PENCEMARAN NAMA BAIK')
        # plt.show()
        st.pyplot()
