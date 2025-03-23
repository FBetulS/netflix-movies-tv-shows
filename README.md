# Netflix Movies and TV Shows Analysis

## 📖 Proje Açıklaması
Bu proje, Netflix'in film ve TV şovlarına dair kapsamlı bir analiz yapmayı amaçlamaktadır. Kullanıcıların içeriklerle etkileşimlerini artırmak, içeriklerin popülaritesini anlamak ve öneri sistemleri geliştirmek için çeşitli görselleştirmeler ve veri analizi teknikleri kullanılmaktadır. Proje, veri kümesindeki içeriklerin türleri, yıllara göre dağılımları, yönetmen ve oyuncu ilişkileri gibi önemli bilgileri ortaya koymayı hedefler.

⚠️ Not
3D grafiklerim ve görselleştirmelerim maalesef gözükmüyor. Bu durum, bazı tarayıcı veya platform uyumsuzluklarından kaynaklanabilir.

## 🔗 Veri Kümesi
Veri kümesi, [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows/data) adresinden alınmıştır. Bu veri kümesi, Netflix'in platformundaki içeriklerin detaylarını içermektedir.

## 🔗 Hugging Face Uygulaması
Ayrıca, projenin etkileşimli bir versiyonu [Netflix Analysis - Hugging Face Space](https://huggingface.co/spaces/btulftma/Netflix-Analysis) adresinde bulunmaktadır.

## 🛠️ Kullanılan Kütüphaneler
- `pandas`: Veri işleme ve analizi için.
- `numpy`: Sayısal işlemler için.
- `matplotlib`: Görselleştirme için.
- `seaborn`: İleri düzey görselleştirme için.
- `wordcloud`: Kelime bulutları oluşturmak için.
- `plotly`: Etkileşimli grafikler için.
- `sklearn`: Makine öğrenimi ve öneri sistemi için.
- `joypy`: Eğlenceli görselleştirmeler için.
- `missingno`: Eksik verilerin görselleştirilmesi için.

## 📈 Analiz Adımları
1. **Veri Yükleme ve Temizlik**:
   - Veri kümesi yüklenir ve gerekli temizlik işlemleri yapılır.
   - Tarih formatları düzeltilir ve eksik değerler doldurulur.

2. **Görselleştirmeler**:
   - Eksik veri haritası.
   - İçerik türlerinin dağılımı.
   - Ülkelere göre içerik dağılımı.
   - Yıllara göre içerik üretimi.
   - Rating dağılımı.
   - Yönetmen ve aktörlerin network grafiği.
   - Türlerin zaman içindeki değişimi.
   - Metin analizi (kelime bulutu).
   - Coğrafi dağılım haritası.

3. **İçerik Öneri Sistemi**:
   - TF-IDF ile içerik açıklamalarından benzerlik hesaplanarak öneri sistemi geliştirilir. 
