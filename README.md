# Ses Transkripsiyon Uygulaması

Bu proje, mikrofon aracılığıyla ses kaydı alıp, Hugging Face Whisper modeli kullanarak transkripsiyon yapan bir Python uygulamasıdır. Uygulama, kullanıcının "q" tuşu ile kaydı durdurmasına olanak tanır. Her kayıt için, seçtiğiniz proje klasörünün altında zaman damgalı bir alt klasör (proje) oluşturulur ve ses dosyası (`audio.wav`) ile transkripsiyon (`transcription.txt`) bu klasöre kaydedilir. Transkripsiyon dili (Türkçe `tr` veya İngilizce `en`) terminalden seçilir ve `config.json` dosyasında saklanarak sonraki çalıştırmalarda hatırlanır. Konsol çıktıları, [Rich](https://rich.readthedocs.io/en/stable/) kütüphanesi kullanılarak kullanıcı dostu hale getirilmiştir.


> **Önemli:** Bu projede, modelin çalışması için **PyTorch** paketi bilgisayarınızda kurulu olmalıdır. Uygulama, transkripsiyon işlemi sırasında GPU yerine CPU kullanır.

---

## Özellikler

- **Interaktif Ses Kaydı:**  
  Kaydı başlatmak için Enter tuşuna basın, kaydı durdurmak için "q" tuşuna basın.

- **Proje Klasörleri:**  
  Başlangıçta kayıtların saklanacağı proje klasörünü onaylarsınız (varsayılan `./recordings`). Her kayıt için bu klasörün altında `YYYY-MM-DD_HH-MM-SS` biçiminde zaman damgalı bir alt klasör oluşturulur; ses dosyası `audio.wav`, transkripsiyon ise `transcription.txt` olarak bu klasöre kaydedilir.

- **Dil Seçimi:**  
  Transkripsiyon dilini terminalden seçersiniz: `tr` (Türkçe) veya `en` (İngilizce). Seçiminiz `config.json` dosyasında saklanır ve siz değiştirene kadar kullanılır. Menüden `l` tuşu ile dili her zaman değiştirebilirsiniz.

- **Başlangıçta Hatırlanan Varsayılanlar:**  
  Her açılışta proje klasörü, dil, mikrofon ve hoparlör (sistem sesi) kaynağı sırayla sorulur ve **en son seçiminiz varsayılan olarak önerilir** — değiştirmek istemiyorsanız sadece `Enter`'a basın. Tüm seçimler `config.json`'da saklanır (cihazlar **isimle**, çünkü cihaz indeksleri oturumlar arasında değişebilir).

- **Mikrofon (Giriş) Seçimi:**  
  Giriş yapabilen ses cihazları (fiziksel mikrofonlar ve `BlackHole`, `ZoomAudioDevice`, `Microsoft Teams Audio` gibi sanal/uygulama cihazları) listelenir ve birini seçersiniz. Menüden `d` tuşu ile her zaman değiştirebilirsiniz.

- **Hoparlör (Sistem Sesi) Kaydı — Mikrofon ile Birlikte:**  
  İsteğe bağlı bir **hoparlör / sistem sesi kaynağı** seçebilirsiniz (varsayılan öneri: `BlackHole`). Seçildiğinde uygulama **mikrofonu ve hoparlör sesini aynı anda** kaydeder ve tek bir mono WAV dosyasında birleştirir (transkripsiyon için yeterli; kaynaklar en kısa olana göre kırpılır, toplanır ve gerekirse kırpılmayı önlemek için ölçeklenir). Menüden `s` tuşu ile bu kaynağı seçebilir veya kapatabilirsiniz; kapalıyken yalnızca mikrofon kaydedilir.  
  *Not: macOS bir çıkış (hoparlör) cihazını doğrudan kaydetmeye izin vermez. Hoparlör/Zoom sesini yakalamak için sistem (veya Zoom) çıkışını bir loopback giriş cihazına (ör. `BlackHole`) yönlendirmiş olmanız gerekir — genelde `BlackHole` + hoparlörünüzü içeren bir **Multi-Output Device** ile, böylece sesi hem duyar hem kaydedersiniz. Bu yönlendirme işletim sistemi tarafında yapılır; uygulama yalnızca cihazı seçip kaydı birleştirir.*

- **Dile Göre Model:**  
  Her dil için en iyi sonucu veren ayrı bir model kullanılır:
  - **Türkçe:** [`selimc/whisper-large-v3-turbo-turkish`](https://huggingface.co/selimc/whisper-large-v3-turbo-turkish) (Hugging Face transformers ile çalışır; ilk kullanımda indirilir).
  - **İngilizce:** [`ggml-distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3-ggml) (whisper.cpp / `pywhispercpp` ile çalışır; ilk kullanımda ~1.5 GB model indirilir).
  - Modeller CUDA / Apple Silicon (MPS/Metal) / CPU arasından uygun olan cihazda çalıştırılır.

- **Kullanıcı Dostu Konsol Çıktıları:**  
  [Rich](https://rich.readthedocs.io/en/stable/) kütüphanesi ile stilize edilmiş paneller, renkli mesajlar ve interaktif promptlar kullanılır.

- **Sürekli Kullanım:**  
  Her kayıttan sonra bir menü gösterilir: `[Enter]` yeni kayıt, `l` dili değiştir, `q` çıkış.

---

## Gereksinimler

- **Python:** 3.10 sürümü gereklidir.
- **pip:** Python paket yöneticisi

### Gerekli Python Paketleri

Bu projede aşağıdaki kütüphaneler kullanılmaktadır:

- `numpy`
- `sounddevice`
- `pynput`
- `transformers`
- `huggingface_hub`
- `pywhispercpp`
- `rich`
- `torch`

Projeyi çalıştırmadan önce bu kütüphaneleri yüklemeniz gerekmektedir. Bunun için `requirements.txt` dosyasını kullanabilirsiniz.

---

## Kurulum Adımları

Bu projeyi sıfırdan çalıştırmak isteyenler için adım adım yapılması gerekenler aşağıdadır:

1. **Repo'yu Klonlayın / İndirin:**  
   Proje dosyalarını bilgisayarınıza indirin veya repoyu klonlayın.

2. **Virtual Environment Oluşturun:**  
   Terminali açın ve proje dizinine gidin. Aşağıdaki komutla sanal ortam oluşturun:
   ```bash
   python3 -m venv .venv

3. **Virtual Environment'ı Aktif Edin:**
   Oluşturduğunuz sanal ortamı şu komut ile aktif edin:
   ```bash
   source .venv/bin/activate
(Windows kullanıyorsanız .\.venv\Scripts\activate komutunu kullanabilirsiniz.)

4. **Python 3.10 Yüklemesi (macOS için):**  
   Eğer macOS kullanıyorsanız ve sisteminizde uygun bir Python sürümü yoksa, Homebrew üzerinden Python 3.10 yükleyebilirsiniz:
   ```bash
   brew install python@3.10

5. **Mimari Kontrolü (Apple Silicon için):**
   Projenin ARM64 mimaride çalıştığından emin olmak isterseniz aşağıdaki komutları çalıştırın:
    ```bash
   python3 -c "import platform; print(platform.machine())"

   ve

    uname -m

Her iki komut da arm64 çıktısı vermelidir. Özellikle Apple Silicon (M1/M2) cihazlarda doğru mimaride çalıştığınız bu şekilde doğrulanır.

6. **Gerekli Paketleri Yükleyin:**  
   Proje dizininde bulunan requirements.txt dosyasını kullanarak gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   
7. **Uygulamayı Çalıştırın:**
Terminalde aşağıdaki komutu kullanarak uygulamayı başlatın (dosya adını kendi script adınıza göre değiştirebilirsiniz):
    ```bash
    python meeting-transcriptor.py

## Kullanım

### Uygulamayı Başlatma:
Terminalde scripti çalıştırdığınızda, "Ses Transkripsiyon Uygulamasına Hoşgeldiniz!" mesajı görüntülenecektir.

### Kayda Başlama:
Kayıt yapmak için Enter tuşuna basın. Kayıt başladıktan sonra, kaydı durdurmak için "q" tuşuna basın.

### Başlangıç (Proje Klasörü, Dil, Mikrofon, Hoparlör):
Uygulama her açıldığında sırayla şunları sorar: proje klasörü, transkripsiyon dili (`tr`/`en`), mikrofon ve hoparlör (sistem sesi) kaynağı. **Her birinde en son seçiminiz varsayılan olarak gelir; korumak için `Enter`'a basın, değiştirmek için yeni değeri girin/seçin.** Tercihler `config.json` dosyasına kaydedilir. **Önemli:** Sisteminizin varsayılan girişi `BlackHole` gibi sanal bir cihazsa ve oraya ses yönlendirilmiyorsa kayıt sessiz olur; bu yüzden mikrofon olarak gerçek mikrofonunuzu seçtiğinizden emin olun.

### Transkripsiyon:
Kayıt durduktan sonra, ses dosyası ilgili proje alt klasörüne `audio.wav` olarak kaydedilir ve seçilen dile uygun model (Türkçe için `selimc/whisper-large-v3-turbo-turkish`, İngilizce için `ggml-distil-large-v3`) transkripsiyon yapar. Sonuç konsolda görüntülenir ve aynı klasöre `transcription.txt` olarak yazılır.

### Menü:
Her kayıttan sonra bir menü gösterilir:

- `[Enter]` — yeni kayıt başlatır.
- `l` — transkripsiyon dilini (`tr`/`en`) değiştirir; tercih hemen `config.json`'a kaydedilir.
- `d` — mikrofon (giriş) cihazını değiştirir; tercih hemen `config.json`'a kaydedilir.
- `s` — hoparlör (sistem sesi) kaynağını seçer veya kapatır; tercih hemen `config.json`'a kaydedilir.
- `q` — uygulamadan çıkar.

### Notlar:
- macOS ortamında "This process is not trusted! Input event monitoring will not be possible until it is added to accessibility clients." uyarısı alabilirsiniz. Bu, sistem erişilebilirlik izinleriyle ilgilidir ve uygulamanın çalışmasını etkilemez. Uyarının görünmemesini istiyorsanız, Terminal veya kullandığınız IDE'yi Erişilebilirlik listesine eklemeniz gerekebilir.