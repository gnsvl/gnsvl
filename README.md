# Sesle Konuşmacı Tanıma — Colab Başlangıç Projesi

Bu proje, **açık rızası olan ve önceden kayıt edilmiş kişiler** arasından bir ses kaydının sahibini tahmin eder. Ses biyometrik veridir; kayıtları izinsiz toplamayın, paylaşmayın veya gerçek hayatta tek başına kesin kimlik kanıtı olarak kullanmayın.

## Hızlı kullanım

1. [speaker_recognition_colab.ipynb](speaker_recognition_colab.ipynb) dosyasını Google Colab'a yükleyin.
2. Aşağıdaki düzende bir `dataset.zip` hazırlayın:

```text
data/
  ayse/
    001.wav
    002.wav
  mehmet/
    001.wav
    002.wav
```

3. Her kayıtlı kişi için mümkünse farklı gün ve ortamlarda alınmış en az 15–20 kısa ses kaydı ekleyin. WAV önerilir; Mac'ten QuickTime ile kaydedilen `.m4a` dosyaları da desteklenir. Her ses 3–10 saniye olması iyi bir başlangıçtır.
4. Not defterindeki hücreleri sırayla çalıştırın. Eğitim sonunda `speaker_model.joblib` indirilebilir.
5. Son hücrede bir test sesini yükleyerek tahmin alın.

Model, hazır ECAPA-TDNN ses modelini ses imzası (embedding) çıkarmak için kullanır; ardından bu imzalarla SVM sınıflandırıcısı eğitir. Bu, küçük veri kümelerinde sıfırdan derin öğrenme eğitmeye göre daha sağlam bir başlangıçtır.

Not: Sistem yalnızca eğitimdeki kayıtlı kişiler arasında seçim yapar. Tanınmayan kişileri güvenilir biçimde ayırmak için sonraki aşamada bir benzerlik eşiği ve "bilinmiyor" sınıfı eklenmelidir.

## Uygulama olarak çalıştırma

Eğitim sonunda indirilen `speaker_model.joblib` dosyasını saklayın. Ardından iki seçeneğiniz vardır:

### Google Colab'da

Not defterinin en altına yeni bir kod hücresi ekleyin ve sırayla şunları çalıştırın:

```python
!pip -q install gradio
!wget -q https://raw.githubusercontent.com/<KULLANICI_ADIN>/<PROJE_ADIN>/main/gradio_app.py
!python gradio_app.py
```

Bu yöntem için önce proje dosyalarını GitHub'a yüklemek gerekir. Yeni başlayanlar için aşağıdaki Mac yöntemi daha kolaydır.

### Mac'te

Terminal'i açın, bu klasöre gidin ve aşağıdaki komutları çalıştırın:

```bash
pip3 install -r requirements.txt
python3 gradio_app.py
```

Ekranda çıkan yerel bağlantıyı tarayıcıda açın. Uygulamada önce `speaker_model.joblib`, sonra test sesini seçip **Sesi kontrol et** düğmesine basın.
