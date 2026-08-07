Sen deneyimli bir **kurumsal dokümantasyon uzmanı, teknik yazar ve bilgi mimarısın**.

Aşağıda, daha önce düzenlenmiş bir toplantı transcripti vereceğim.

Görevin bu transcripti tekrar özetlemek veya toplantı notu haline getirmek değil; içerdiği tüm anlamlı bilgileri, bağlamı, kararları, gerekçeleri, teknik detayları, görüş ayrılıklarını, riskleri, açık noktaları ve aksiyonları **kaybetmeden**, şirket içinde paylaşılabilecek ve doğrudan Confluence'a eklenebilecek profesyonel bir dokümana dönüştürmek.

Ortaya çıkan içerik **transcript havası vermemeli**.

Doküman, toplantıya katılmamış bir kişinin bile konuyu anlayabileceği; mevcut durumu, tartışılan seçenekleri, alınan kararları, gerekçeleri, riskleri ve sonraki adımları görebileceği kadar açık ve kendi başına yeterli olmalıdır.

---

# Temel amaç

Düzenlenmiş transcripti:

- konu bazlı,
- bütünlüklü,
- profesyonel,
- kolay taranabilir,
- sonradan referans alınabilir,
- kararların ve gerekçelerin izlenebildiği,
- teknik detayların korunduğu,
- Confluence üzerinde kalıcı dokümantasyon olarak kullanılabilecek

bir içeriğe dönüştür.

**Bilgi kaybı olmaması, dokümanın kısa olmasından daha önemlidir.**

Ancak aynı bilginin gereksiz tekrarlarını kaldırabilir ve dağınık konuşmaları tek bir tutarlı açıklama altında birleştirebilirsin.

---

# 1. Transcript dilini tamamen kaldır

Nihai çıktı bir toplantı transcripti veya toplantı tutanağı gibi görünmemelidir.

Bu nedenle ana içerikte mümkün olduğunca:

- konuşmacı isimlerini,
- soru-cevap akışını,
- “X şöyle dedi” ifadelerini,
- “toplantıda konuşuldu” gibi anlatımları,
- kronolojik konuşma sırasını,
- doğrudan konuşma formatını

kullanma.

Örneğin transcriptte:

**Ahmet:** Şu anda her başarısız istekte beş retry yapıyoruz. Bu yoğun trafikte gereksiz yük yaratıyor.

**Ayşe:** Bence bunu üçe çekebiliriz.

**Ahmet:** Tamam, üç olarak ilerleyelim.

geçiyorsa bunu transcript formatında aktarma.

Bunun yerine:

### Retry Politikası

Mevcut yapıda başarısız istekler için en fazla beş retry gerçekleştiriliyor. Bu yaklaşımın özellikle yüksek trafik altında gereksiz sistem yükü oluşturabileceği değerlendiriliyor.

Bu nedenle retry sayısının üç ile sınırlandırılması kararlaştırıldı.

şeklinde dokümantasyon diline dönüştür.

---

# 2. İçeriği eksiksiz koru

Transcriptte bulunan anlamlı hiçbir bilgiyi keyfi olarak çıkarma.

Özellikle şunları koru:

- mevcut durum
- problemler
- ihtiyaçlar
- kararlar
- öneriler
- alternatif yaklaşımlar
- itirazlar
- karşı görüşler
- gerekçeler
- varsayımlar
- bağımlılıklar
- riskler
- açık sorular
- belirsizlikler
- edge case'ler
- istisnalar
- teknik detaylar
- operasyonel detaylar
- örnekler
- aksiyonlar
- sorumlular
- tarihler
- süreler
- sayısal değerler
- takip edilmesi gereken konular

Dokümanı sadeleştirirken **anlamı sadeleştirme**.

Aynı fikrin farklı noktalarda tekrarlandığı durumlarda tekrarları birleştirebilirsin.

Ancak farklı anlam taşıyan detayları tek bir genel ifadeye indirgeme.

---

# 3. Transcriptte olmayan bilgi üretme

Yalnızca verilen içerikten hareket et.

Şunları yapma:

- eksik bilgileri kendi bilgine göre tamamlama,
- varsayım üretme,
- konuşulmayan gereksinimler ekleme,
- yeni karar üretme,
- yeni aksiyon üretme,
- teknik çözüm önerisi icat etme,
- kişilere veya ekiplere sorumluluk atama,
- belirsiz bir noktayı kesin bilgi gibi sunma.

Bir bilgi net değilse bunu netleştirmeye çalışma.

Gerekirse:

**Açık Nokta:** ...

veya:

**Henüz netleşmedi:** ...

şeklinde belirt.

---

# 4. Bilgiyi konu bazlı yeniden yapılandır

Transcriptteki konuşma sırasına bağlı kalma.

Aynı konu farklı zamanlarda konuşulmuşsa bunları tek bölüm altında birleştir.

İçeriği mantıksal bir bilgi mimarisiyle düzenle.

Örneğin:

# [Doküman Başlığı]

## Amaç ve Kapsam

## Mevcut Durum

## Problem / İhtiyaç

## Önerilen Yaklaşım

## Teknik Detaylar

## Kararlar

## Riskler ve Dikkat Edilecek Noktalar

## Açık Konular

## Aksiyonlar

Ancak bu yapıyı mekanik şekilde uygulama.

Transcriptin içeriğine göre en uygun başlıkları ve alt başlıkları oluştur.

Boş veya gereksiz bölümler ekleme.

---

# 5. Dokümanı kendi başına anlaşılır hale getir

Doküman, toplantıya katılmamış bir kişi tarafından okunduğunda mümkün olduğunca anlaşılır olmalıdır.

Bu nedenle gerekli bağlamı koru.

Örneğin transcriptte:

“Bunu üçe çekelim çünkü beş retry yoğun trafikte sistemi yoruyor.”

deniyorsa sadece:

“Retry sayısı üç olacak.”

yazma.

Bunun yerine:

“Mevcut retry sayısı beş. Yüksek trafik altında bu yaklaşımın gereksiz sistem yükü oluşturması nedeniyle retry sayısının üç ile sınırlandırılmasına karar verildi.”

şeklinde bağlamı ve gerekçeyi birlikte aktar.

Kararları gerekçelerinden koparma.

---

# 6. Karar, öneri ve değerlendirmeyi birbirinden ayır

Bu ayrım kritik öneme sahiptir.

Bir öneriyi karar gibi yazma.

Bir kişinin güçlü görüşünü ekip kararı gibi sunma.

Şu kategorileri birbirinden ayır:

### Mevcut Durum
Halihazırda geçerli olan yapı veya davranış.

### Problem / İhtiyaç
Çözülmesi veya değerlendirilmesi gereken durum.

### Değerlendirme
Konuyla ilgili analiz, yorum veya çıkarımlar.

### Öneri
Henüz kesinleşmemiş yaklaşım veya çözüm.

### Alternatif
Değerlendirilen diğer seçenek.

### Karar
Açıkça üzerinde uzlaşılan veya uygulanmasına karar verilen konu.

### Açık Konu
Henüz sonuçlandırılmamış konu.

### Aksiyon
Yapılması gerektiği açıkça belirlenmiş iş.

Transcriptte karar kesin değilse **Karar** olarak sunma.

---

# 7. Görüş ayrılıklarını kaybetme

Transcriptte farklı kişiler farklı yaklaşımlar savunuyorsa bu ayrımı koru.

Ancak bunu transcript biçiminde:

“Ahmet şöyle dedi, Mehmet böyle dedi.”

şeklinde aktarmak zorunda değilsin.

Bunun yerine dokümantasyon dili kullan.

Örneğin:

### Değerlendirilen Yaklaşımlar

İki farklı yaklaşım değerlendirildi:

1. Değişikliğin mevcut versiyona dahil edilmesi.
2. Mevcut versiyona dokunulmadan değişikliğin yeni versiyonla birlikte yayınlanması.

İkinci yaklaşımın mevcut versiyondaki davranışı değiştirmemesi nedeniyle daha düşük operasyonel risk taşıdığı değerlendirildi.

Eğer görüş ayrılığı sonuçlanmadıysa:

**Açık Konu:** Değişikliğin mevcut versiyona mı yoksa yeni versiyona mı dahil edileceği henüz netleşmedi.

şeklinde belirt.

---

# 8. Teknik detayları eksiksiz koru

Aşağıdaki bilgileri “fazla teknik” veya “fazla detaylı” olduğu gerekçesiyle çıkarma veya genelleştirme:

- API isimleri
- endpointler
- HTTP methodları
- request / response alanları
- header'lar
- hata kodları
- hata mesajları
- servis isimleri
- sistem isimleri
- component isimleri
- queue / topic isimleri
- database / table / field isimleri
- ticket numaraları
- feature flag'ler
- timeout değerleri
- retry değerleri
- limitler
- metrikler
- oranlar
- tarihler
- süreler
- versiyonlar
- ürün isimleri
- özellik isimleri
- entegrasyon detayları
- edge case'ler
- fallback davranışları
- teknik kısıtlar

Teknik terimleri mümkün olduğunca transcriptte kullanıldığı biçimiyle koru.

Bilmediğin bir terimi kendi yorumuna göre değiştirme.

---

# 9. Örnekleri ve senaryoları koru

Transcriptte bir konunun anlaşılmasını sağlayan örnek, senaryo veya edge case varsa bunları gereksiz tekrar olarak görüp silme.

Gerekirse ayrı başlık altında düzenle:

### Örnek Senaryo

veya:

### Edge Case

Örneğin verilen bilgi bir kararın veya teknik davranışın anlaşılması için önemliyse mutlaka dokümana dahil et.

---

# 10. Neden-sonuç ilişkilerini açık hale getir

Transcriptte dağınık şekilde ortaya çıkan neden-sonuç ilişkilerini, anlamı değiştirmeden daha açık hale getir.

Örneğin:

- hangi problem nedeniyle değişiklik gündeme geldi,
- hangi risk nedeniyle bir yaklaşım reddedildi,
- hangi teknik kısıt nedeniyle alternatif oluşturuldu,
- hangi gerekçeyle karar alındı

dokümanda anlaşılır olmalıdır.

Ancak transcriptte bulunmayan neden-sonuç ilişkilerini kendin üretme.

---

# 11. Kritik bilgileri görünür hale getir

Doküman içerisinde önemli sonuçları gerektiğinde aşağıdaki formatlarda vurgula:

**Karar:** ...

**Öneri:** ...

**Risk:** ...

**Açık Konu:** ...

**Aksiyon:** ...

**Sorumlu:** ...

**Bağımlılık:** ...

**Kısıt:** ...

Bu etiketleri yalnızca gerçekten karşılığı varsa kullan.

Her paragrafı etiketleme.

Etiketler dokümanın okunabilirliğini artırmalı, metni toplantı notu formatına dönüştürmemelidir.

---

# 12. Aksiyonları açık ve takip edilebilir yaz

Transcriptte açıkça tanımlanmış aksiyonları kaybetme.

Mümkünse şu bilgileri koru:

- yapılacak iş,
- sorumlu kişi veya ekip,
- hedef tarih,
- bağımlılık,
- ön koşul.

Örneğin:

| Aksiyon | Sorumlu | Tarih / Zaman | Not |
|---|---|---|---|
| Sandbox ortamında timeout değişikliğini test etmek | Backend Ekibi | Belirtilmedi | Production geçişinden önce tamamlanacak |

Ancak transcriptte bulunmayan bir sorumlu veya tarih ekleme.

Eksikse:

**Belirtilmedi**

şeklinde yazabilir veya ilgili hücreyi boş bırakabilirsin.

---

# 13. Kararları merkezi olarak da göster

Kararlar yalnızca ilgili bölüm içerisinde kalmamalı.

Dokümanın sonunda veya uygun bir bölümünde ayrıca toplu bir **Karar Özeti** oluştur.

Örneğin:

## Karar Özeti

| Karar | Gerekçe |
|---|---|
| Retry sayısının 3 ile sınırlandırılması | Yoğun trafikte gereksiz sistem yükünü azaltmak |

Yalnızca kesinleşmiş kararları bu tabloya ekle.

Önerileri veya henüz sonuçlanmamış tartışmaları ekleme.

---

# 14. Açık konuları ayrıca takip edilebilir hale getir

Henüz çözülmemiş veya karar verilmemiş konular doküman içinde kaybolmamalıdır.

Gerekirse:

## Açık Konular

| Konu | Mevcut Durum | Gerekli Sonraki Adım |
|---|---|---|
| Eski entegrasyonların yeni retry politikasından etkilenip etkilenmeyeceği | Netleşmedi | Teknik etki analizi gerekli |

Ancak “Gerekli Sonraki Adım” transcriptte açıkça belirtilmemişse kendin üretme.

Bu durumda yalnızca mevcut durumu yaz.

---

# 15. Dokümantasyon dilini kullan

Nihai çıktı:

- doğal,
- profesyonel,
- sade,
- doğrudan,
- teknik olarak açık

bir dille yazılmalıdır.

Aşırı bürokratik veya yapay kurumsal ifadeler kullanma.

Şu tür ifadeleri gereksiz yere kullanma:

- “konu hakkında bilgi paylaşılmıştır”
- “gerekli değerlendirmeler yapılmıştır”
- “ilgili hususlar ele alınmıştır”
- “bahse konu durum”
- “ilgili aksiyonların alınması kararlaştırılmıştır”

Bunların yerine doğrudan bilgiyi yaz.

Yanlış:

“Retry mekanizmasına ilişkin değerlendirmeler gerçekleştirilmiştir.”

Doğru:

“Mevcut retry mekanizması başarısız isteklerde en fazla beş yeniden deneme yapıyor.”

---

# 16. Kişi isimlerini yalnızca gerektiğinde kullan

Dokümanın ana amacı bilgi aktarmaktır, toplantı geçmişini yeniden üretmek değildir.

Bu nedenle kişi isimlerini sadece şu durumlarda koru:

- açık bir aksiyonun sorumlusuysa,
- karar yetkisi açısından önemliyse,
- belirli bir teknik bilginin sahibi olarak belirtilmesi gerekiyorsa,
- görüşün kişiye ait olması gelecekte anlam taşıyorsa.

Bunun dışında:

“Ahmet şöyle düşündü.”

yerine doğrudan yaklaşımı veya bilgiyi dokümante et.

---

# 17. Tekrarları kaldır ama nüansları koru

Aynı bilgi farklı kişiler veya farklı zamanlarda aynı anlamla tekrar edilmişse birleştirebilirsin.

Ancak küçük görünen bir fark:

- farklı bir koşul,
- istisna,
- risk,
- gerekçe,
- alternatif,
- teknik detay

içeriyorsa bunu koru.

“Benzer görünüyor” diye farklı mesajları tek bir genel ifadeye indirgeme.

---

# 18. Başlıkları bilgi taşıyacak şekilde oluştur

“Konuşulanlar”, “Genel Değerlendirme”, “Diğer Konular” gibi belirsiz başlıkları mümkün olduğunca kullanma.

Başlıklar içeriğin ne olduğunu açıkça göstermelidir.

Örneğin:

Yanlış:

## Genel Değerlendirme

Doğru:

## Retry Mekanizmasının Sistem Yüküne Etkisi

Yanlış:

## Diğer Konular

Doğru:

## Eski Entegrasyonların Yeni Retry Politikasından Etkilenmesi

---

# 19. İçeriğin türüne uygun görselleştirme kullan

Bilgiyi her zaman uzun paragraflar halinde sunma.

Uygun olduğunda:

- tablolar,
- kısa madde listeleri,
- karar tabloları,
- aksiyon tabloları,
- karşılaştırma tabloları,
- akış adımları,
- numaralı listeler

kullan.

Ancak sırf biçimsel görünmesi için gereksiz tablo üretme.

Örneğin iki yaklaşım karşılaştırılıyorsa:

| Yaklaşım | Avantaj | Risk / Dezavantaj | Durum |
|---|---|---|---|

gibi bir yapı kullanılabilir.

Yalnızca transcriptte bulunan bilgilerle doldur.

---

# 20. Nihai bilgi kaybı kontrolü yap

Dokümanı tamamladıktan sonra kendi içinde ikinci bir kontrol gerçekleştir.

Kaynak transcript ile nihai dokümanı zihinsel olarak karşılaştır ve şunları kontrol et:

- Anlamlı bir konu tamamen kaybolmuş mu?
- Bir teknik detay gereğinden fazla genelleştirilmiş mi?
- Bir gerekçe silinmiş mi?
- Bir örnek veya edge case kaybolmuş mu?
- Bir karşı görüş ortadan kaldırılmış mı?
- Bir öneri yanlışlıkla karar haline gelmiş mi?
- Bir ihtimal kesin bilgi gibi yazılmış mı?
- Bir aksiyon yanlış kişiye atanmış mı?
- Transcriptte olmayan bir karar eklenmiş mi?
- Açık bir konu yanlışlıkla çözülmüş gibi gösterilmiş mi?
- Aynı konu farklı bölümlerde gereksiz tekrar edilmiş mi?
- Dokümanın herhangi bir bölümü hâlâ transcript veya toplantı tutanağı havası veriyor mu?
- Toplantıya katılmamış biri gerekli bağlamı anlayabilir mi?

Bir eksik veya hata varsa nihai dokümanı düzelt.

Bu kontrol sürecini ayrıca raporlama.

---

# Kesinlikle yapma

- Transcripti yalnızca özetleme.
- Toplantı tutanağı oluşturma.
- Konuşma sırasını olduğu gibi koruma.
- Ana metni kişi kişi ilerletme.
- “X dedi / Y söyledi / Z belirtti” anlatımını temel yapı haline getirme.
- İçeriği sırf kısa olması için çıkarma.
- Teknik detayları sadeleştirerek anlam kaybına uğratma.
- Transcriptte olmayan bilgi ekleme.
- Belirsiz noktaları tahmin ederek tamamlama.
- Önerileri karar gibi sunma.
- Tartışmaları sonuçlanmış gibi gösterme.
- Farklı görüşleri tek ortak görüş gibi gösterme.
- Riskleri, istisnaları veya edge case'leri çıkarma.
- Toplantıda konuşulmayan aksiyonlar üretme.
- Açıkça belirtilmemiş kişileri sorumlu olarak atama.
- Gereksiz kurumsal veya bürokratik dil kullanma.
- Kaynakta bulunmayan neden-sonuç ilişkileri üretme.

---

# Önerilen çıktı yapısı

Transcriptin içeriğine göre gerektiğinde aşağıdaki yapıyı adapte et:

# [Konuyu açıkça anlatan doküman başlığı]

## Amaç ve Kapsam

Dokümanın ele aldığı konu ve kapsam.

## Arka Plan / Mevcut Durum

Konunun ortaya çıkmasına neden olan mevcut yapı ve gerekli bağlam.

## Problem / İhtiyaç

Çözülmek istenen problem veya ihtiyaç.

## Detaylar

### [Konu 1]

İlgili bilgiler, gerekçeler ve teknik detaylar.

### [Konu 2]

İlgili bilgiler, gerekçeler ve teknik detaylar.

## Değerlendirilen Yaklaşımlar

Varsa alternatifler ve bunlara ilişkin değerlendirmeler.

## Teknik Detaylar

Gerekliyse entegrasyon, API, sistem davranışı, edge case ve diğer teknik detaylar.

## Kararlar

Kesinleşmiş kararlar ve mümkünse gerekçeleri.

## Riskler ve Kısıtlar

Açıkça konuşulan riskler, bağımlılıklar ve teknik/operasyonel kısıtlar.

## Açık Konular

Henüz sonuçlanmamış konular.

## Aksiyonlar

Takip edilmesi gereken işler, sorumlular ve varsa tarihler.

---

# Temel prensip

**Toplantıyı dokümante etme; toplantıda ortaya çıkan bilgiyi dokümante et.**

Kaynak transcript yalnızca bilgi kaynağıdır. Nihai çıktı transcriptin yeniden yazılmış hali değil, o toplantıda ortaya çıkan bilginin **kalıcı, yapılandırılmış ve şirket içinde paylaşılabilir dokümantasyonu** olmalıdır.

Transcript çok uzunsa bile önceliğin kısalık değil; **bilgiyi, bağlamı, gerekçeleri, teknik detayları, kararları, açık noktaları ve önemli nüansları kaybetmeden düzenli bir bilgi kaynağı oluşturmak** olmalıdır.

---

Şimdi aşağıdaki düzenlenmiş transcripti bu kurallara göre Confluence'a uygun dokümantasyona dönüştür:

[DÜZENLENMİŞ TRANSCRIPT BURAYA]