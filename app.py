import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import missingno as msno
from collections import Counter

# Sayfa yapılandırması
st.set_page_config(layout="wide", page_title="Netflix Veri Analizi", page_icon=":movie_camera:")

# Başlık ve açıklama
st.title("Netflix İçerik Analizi Uygulaması")
st.markdown("Bu uygulama, Netflix veri setini analiz eder ve çeşitli görselleştirmeler sunar.")

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['duration'] = df['duration'].str.extract('(\d+)').astype(float)
    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['country'] = df['country'].fillna('Unknown').str.split(', ')
    df['genre'] = df['listed_in'].str.split(', ')
    return df

# Veri yükleme
try:
    df = load_data()
    st.success("Veri başarıyla yüklendi!")
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.info("Lütfen 'netflix_titles.csv' dosyasının doğru konumda olduğundan emin olun.")
    st.stop()

# Sidebar menüsü
st.sidebar.title("Analiz Seçenekleri")
analysis_option = st.sidebar.selectbox(
    "Hangi analizi görmek istersiniz?",
    ["Veri Özeti", "Eksik Veri Analizi", "İçerik Türü Dağılımı", 
     "Ülkelere Göre İçerik Dağılımı", "Yıllara Göre İçerik Üretimi", 
     "Rating Dağılımı", "En Üretken Yönetmenler ve Oyuncular", 
     "Türlerin Zaman İçinde Değişimi", "Kelime Bulutu", 
     "Coğrafi İçerik Dağılımı", "İçerik Öneri Sistemi"]
)

# Özel Renk Paleti
netflix_colors = ['#221f1f', '#b20710', '#e50914', '#f5f5f1']

# Veri Özeti
if analysis_option == "Veri Özeti":
    st.header("Veri Seti Özeti")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Veri şekli")
        st.write(f"Satır sayısı: {df.shape[0]}")
        st.write(f"Sütun sayısı: {df.shape[1]}")
        
    with col2:
        st.subheader("İçerik türleri")
        st.write(df['type'].value_counts())
    
    st.subheader("Örnek veriler")
    st.dataframe(df.head())
    
    st.subheader("İstatistiksel Özet")
    st.dataframe(df.describe())

# Eksik Veri Analizi
elif analysis_option == "Eksik Veri Analizi":
    st.header("Eksik Veri Analizi")
    
    # Eksik değer sayıları
    missing_data = pd.DataFrame({'Eksik Değer Sayısı': df.isnull().sum(),
                               'Eksik Değer Yüzdesi': df.isnull().sum() / len(df) * 100})
    
    st.dataframe(missing_data)
    
    # Eksik veri görselleştirmesi
    fig, ax = plt.subplots(figsize=(12, 6))
    msno.matrix(df, color=(0.7, 0.1, 0.1), sparkline=False, ax=ax)
    plt.title('Eksik Veri Dağılımı', fontsize=18, pad=20)
    st.pyplot(fig)

# İçerik Türü Dağılımı
elif analysis_option == "İçerik Türü Dağılımı":
    st.header("İçerik Türü Dağılımı")
    
    types = df['type'].value_counts()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    
    fig.add_trace(go.Pie(labels=types.index, values=types.values, 
                         hole=.4, marker_colors=netflix_colors), 1, 1)
    fig.add_trace(go.Pie(labels=types.index, values=types.values, 
                         hole=.7, showlegend=False), 1, 2)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title_text='İçerik Tür Dağılımı', title_x=0.5)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"Netflix veri setindeki toplam {types.sum()} içeriğin {types['Movie']} tanesi film (%{(types['Movie']/types.sum()*100):.1f}), "
             f"{types['TV Show']} tanesi TV şovu (%{(types['TV Show']/types.sum()*100):.1f}).")

# Ülkelere Göre İçerik Dağılımı
elif analysis_option == "Ülkelere Göre İçerik Dağılımı":
    st.header("Ülkelere Göre İçerik Dağılımı")
    
    country_counts = df.explode('country').groupby('country').size().reset_index(name='count')
    top_countries = country_counts.sort_values('count', ascending=False).head(20)
    
    fig = px.treemap(top_countries, 
                     path=['country'], 
                     values='count',
                     color='count',
                     color_continuous_scale='Reds',
                     title='Ülkelere Göre İçerik Dağılımı')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart for top 10 countries
    top10_countries = top_countries.head(10)
    fig = px.bar(top10_countries, x='country', y='count',
                color='count', color_continuous_scale='Reds',
                title='En Çok İçerik Üreten 10 Ülke')
    
    st.plotly_chart(fig, use_container_width=True)

# Yıllara Göre İçerik Üretimi
elif analysis_option == "Yıllara Göre İçerik Üretimi":
    st.header("Yıllara Göre İçerik Üretimi")
    
    yearly_content = df.groupby(['release_year', 'type']).size().unstack().fillna(0)
    
    fig = px.area(yearly_content, 
                  x=yearly_content.index, 
                  y=yearly_content.columns,
                  title='Yıllara Göre İçerik Üretimi',
                  labels={'value':'İçerik Sayısı', 'release_year':'Yıl'},
                  color_discrete_sequence=['#e50914', '#221f1f'])
    fig.update_layout(hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yıl seçimine göre detaylı analiz
    min_year = int(df['release_year'].min())
    max_year = int(df['release_year'].max())
    
    year_range = st.slider('Yıl aralığı seçin:', min_year, max_year, (min_year, max_year))
    
    filtered_df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]
    year_type_count = filtered_df.groupby(['release_year', 'type']).size().unstack().fillna(0)
    
    st.subheader(f"{year_range[0]}-{year_range[1]} Yılları Arası İçerik Dağılımı")
    st.dataframe(year_type_count)

# Rating Dağılımı
elif analysis_option == "Rating Dağılımı":
    st.header("Rating Dağılımı")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(x='rating', y='release_year', data=df, 
                  inner='quartile', palette='Reds', cut=0, ax=ax)
    plt.title('Ratinglere Göre Yayın Yılı Dağılımı', fontsize=16)
    plt.xlabel('Rating')
    plt.ylabel('Yayın Yılı')
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Rating dağılımı pasta grafiği
    rating_counts = df['rating'].value_counts().sort_values(ascending=False)
    
    fig = px.pie(values=rating_counts.values, names=rating_counts.index,
                title='Rating Dağılımı',
                color_discrete_sequence=px.colors.sequential.Reds)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)

# En Üretken Yönetmenler ve Oyuncular
elif analysis_option == "En Üretken Yönetmenler ve Oyuncular":
    st.header("En Üretken Yönetmenler ve Oyuncular")
    
    # Filtrele: "Unknown" yönetmenleri çıkar
    valid_directors = df[df['director'] != 'Unknown']
    directors = valid_directors['director'].str.split(', ').explode().value_counts().head(20)
    
    # Filtrele: "Unknown" oyuncuları çıkar
    valid_actors = df[df['cast'] != 'Unknown']
    actors = valid_actors['cast'].str.split(', ').explode().value_counts().head(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("En Üretken 20 Yönetmen")
        fig = px.bar(y=directors.index, x=directors.values, orientation='h',
                    color=directors.values, color_continuous_scale='Reds',
                    labels={'x': 'İçerik Sayısı', 'y': 'Yönetmen'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("En Üretken 20 Oyuncu")
        fig = px.bar(y=actors.index, x=actors.values, orientation='h',
                    color=actors.values, color_continuous_scale='Reds',
                    labels={'x': 'İçerik Sayısı', 'y': 'Oyuncu'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# Türlerin Zaman İçinde Değişimi
elif analysis_option == "Türlerin Zaman İçinde Değişimi":
    st.header("Türlerin Zaman İçinde Değişimi")
    
    genre_yearly = df.explode('genre').groupby(['release_year', 'genre']).size().reset_index(name='count')
    top_genres = genre_yearly.groupby('genre')['count'].sum().nlargest(10).index.tolist()
    
    # Çok select box ile türleri seçme
    selected_genres = st.multiselect(
        "Görmek istediğiniz türleri seçin:",
        options=top_genres,
        default=top_genres[:5]
    )
    
    if selected_genres:
        filtered_genre_yearly = genre_yearly[genre_yearly['genre'].isin(selected_genres)]
        
        fig = px.line(filtered_genre_yearly, x='release_year', y='count', color='genre',
                     title='Türlerin Yıllara Göre Değişimi',
                     labels={'count': 'İçerik Sayısı', 'release_year': 'Yıl', 'genre': 'Tür'},
                     line_shape='spline', render_mode='svg')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tür bazında yıllık büyüme
        pivot_genre = filtered_genre_yearly.pivot(index='release_year', columns='genre', values='count').fillna(0)
        st.subheader("Seçili Türlerin Yıllara Göre İçerik Sayıları")
        st.dataframe(pivot_genre)
    else:
        st.warning("Lütfen en az bir tür seçin.")

# Kelime Bulutu
elif analysis_option == "Kelime Bulutu":
    st.header("İçerik Açıklamaları Kelime Bulutu")
    
    text = ' '.join(df['description'].dropna())
    
    wordcloud = WordCloud(width=1600, height=800, 
                          background_color='white',
                          colormap='Reds').generate(text)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('İçerik Açıklamaları Kelime Bulutu', fontsize=20, pad=20)
    
    st.pyplot(fig)
    
    # En çok geçen kelimeler
    st.subheader("Açıklamalarda En Çok Geçen Kelimeler")
    
    # Kelimeleri tokenize etme (basit bir yaklaşım)
    words = text.lower().split()
    # Stopwords (basit bir liste)
    stopwords = ['the', 'a', 'and', 'to', 'of', 'in', 'is', 'for', 'on', 'with', 'his', 'her', 'their', 'when', 'who', 'what', 'how']
    
    # Stopwords olmayan kelimeler
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words).most_common(20)
    
    # DataFrame'e çevirme
    word_df = pd.DataFrame(word_counts, columns=['Kelime', 'Sayı'])
    
    # Bar grafiği
    fig = px.bar(word_df, x='Kelime', y='Sayı', color='Sayı',
                color_continuous_scale='Reds',
                title='Açıklamalarda En Çok Geçen 20 Kelime')
    
    st.plotly_chart(fig, use_container_width=True)

# Coğrafi İçerik Dağılımı
elif analysis_option == "Coğrafi İçerik Dağılımı":
    st.header("Coğrafi İçerik Dağılımı")
    
    country_counts = df.explode('country').groupby('country').size().reset_index(name='count')
    country_geo = country_counts.rename(columns={'country':'name'})
    
    fig = px.choropleth(country_geo, 
                        locations="name",
                        locationmode='country names',
                        color="count", 
                        hover_name="name",
                        color_continuous_scale='Reds',
                        title='Ülkelere Göre İçerik Dağılımı')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Ülkeye göre filtreleme
    top_countries = country_counts.sort_values('count', ascending=False).head(30)['country'].tolist()
    
    selected_country = st.selectbox("Bir ülke seçin:", options=["Tümü"] + top_countries)
    
    if selected_country != "Tümü":
        country_df = df[df['country'].apply(lambda x: selected_country in x)]
        
        st.subheader(f"{selected_country} Ülkesine Ait İçerikler")
        st.write(f"Toplam {len(country_df)} içerik bulundu.")
        
        # Film-Dizi dağılımı
        type_counts = country_df['type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title=f"{selected_country} İçerik Türü Dağılımı",
                        color_discrete_sequence=px.colors.sequential.Reds)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Yıllara göre içerik
            yearly_country = country_df.groupby('release_year').size().reset_index(name='count')
            
            fig = px.bar(yearly_country, x='release_year', y='count',
                        title=f"{selected_country} Yıllara Göre İçerik Sayısı",
                        color='count', color_continuous_scale='Reds')
            
            st.plotly_chart(fig, use_container_width=True)

# İçerik Öneri Sistemi
elif analysis_option == "İçerik Öneri Sistemi":
    st.header("İçerik Öneri Sistemi")
    
    # TF-IDF ve Cosine Similarity hesaplama
    @st.cache_resource
    def create_recommendation_model():
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['description'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        return cosine_sim, indices
    
    cosine_sim, indices = create_recommendation_model()
    
    # Öneri fonksiyonu
    def netflix_recommender(title, num_recommendations=5):
        try:
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations+1]
            movie_indices = [i[0] for i in sim_scores]
            return df[['title', 'type', 'listed_in', 'description']].iloc[movie_indices]
        except:
            return None
    
    # Kullanıcı girdisi
    content_titles = sorted(df['title'].unique())
    selected_title = st.selectbox("İçerik seçin:", options=content_titles)
    
    num_recommendations = st.slider("Kaç öneri görmek istersiniz?", 1, 20, 5)
    
    if st.button("Önerileri Göster"):
        recommendations = netflix_recommender(selected_title, num_recommendations)
        
        if recommendations is not None:
            st.subheader(f"'{selected_title}' için {num_recommendations} Öneri:")
            
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with st.expander(f"{i+1}. {row['title']} ({row['type']})"):
                    st.write(f"**Türler:** {row['listed_in']}")
                    st.write(f"**Açıklama:** {row['description']}")
        else:
            st.error("Bu içerik için öneri yapılamadı veya yeterli veri yok.")
    
    # Rastgele öneri
    if st.button("Rastgele İçerik Öner"):
        random_titles = df['title'].sample(num_recommendations).tolist()
        
        st.subheader(f"Rastgele {num_recommendations} İçerik Önerisi:")
        
        for i, title in enumerate(random_titles):
            content = df[df['title'] == title].iloc[0]
            with st.expander(f"{i+1}. {content['title']} ({content['type']})"):
                st.write(f"**Türler:** {content['listed_in']}")
                st.write(f"**Açıklama:** {content['description']}")

# Footer
st.markdown("---")
st.markdown("Netflix Veri Analizi | Streamlit Uygulaması")
