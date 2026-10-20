# ARCHITECTURE.md — CMP4509 Term Project

Makale: Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization* (arXiv:1611.03530, ICLR 2017). Sign-up listesinde #6.

Bu dosyadaki kararlar kilitlidir. Sapmak gerekirse önce grupta konuşulur, sonra buraya yazılır.

## 1. Makalenin iddiası (tek paragraf)

Standart ağlar (CNN, MLP) CIFAR-10'da **rastgele etiketleri bile train'de neredeyse %100 ezberleyebilir**. Demek ki "ağ genelleme yaptığı için iyi" açıklaması tek başına yetmez: kapasite her şeyi ezberlemeye yetiyor, genelleme başka bir şeyden geliyor. Ayrıca weight decay, dropout, data augmentation gibi açık regularizer'lar ne gerekli ne de yeterli.

## 2. Neyi yeniden üretiyoruz

| Kod | Deney | Makaledeki yeri | Ne gösteriyor |
|---|---|---|---|
| E1 | Randomization testleri: gerçek etiket, rastgele etiket, karıştırılmış piksel, rastgele piksel, Gauss gürültüsü | Fig 1a, 1b | Hepsi train'de fit edilir; rastgele olanlar sadece daha geç yakınsar |
| E2 | Etiket bozulma oranı süpürmesi (0, 0.2, 0.4, 0.6, 0.8, 1.0) | Fig 1c | Bozulma arttıkça test hatası düzgünce artar |
| E3 | Açık regularizer'lar açık/kapalı | Tablo 1 | Regularizer olmadan da genelleme olabilir; regularizer ile de rastgele etiket ezberlenebilir |
| E4 | Theorem 1'in yapıcı ispatı (kapılı uzantı, bkz. §7) | Bölüm 4 | 2n+d parametreli ağ, n örneğe her etiketi fit eder |

Hedef "makaledeki sayıları tutturmak" değil. İddianın bizim ölçeğimizde tutup tutmadığını, tutmuyorsa nedenini raporlamak. Her ölçek küçültmesi raporda belirtilir.

## 3. Kilitlenen teknik kararlar

- **Dil/çatı:** PyTorch.
- **Veri:** CIFAR-10. Tamamı GPU/cihaz belleğine yüklenir, DataLoader kullanılmaz.
- **Modeller:** (a) küçük, BatchNorm'lu CNN (Inception/ResNet benzeri, tam Inception değil), (b) 1 gizli katmanlı MLP (512 birim).
- **Seed:** her konfigürasyon en az 3 seed. Raporda mean ± std.
- **Cihaz:** kodda tek yerde seçilir: `cuda` → `mps` → `cpu`. Başka yerde cihaz adı yazılmaz.
- **Geliştirme:** lokal. **Teslim edilen notebook'lar** Colab veya Kaggle'da baştan sona çalışmak zorunda (duyuru şartı). Final koşuları orada doğrulanır.
- **Checkpoint:** düzenli aralıkla kaydedilir, koşu kaldığı yerden devam edebilir.
- **Raporlama:** her koşunun sonuçları (epoch bazlı train/test loss ve accuracy) bir CSV/JSON'a yazılır. Şekiller bu dosyalardan üretilir, elle sayı girilmez.
- **Config:** her deneyin parametreleri (model, optimizer, lr, epoch, bozulma oranı, seed) tek bir config nesnesinden gelir. Aynı parametre iki yerde tanımlanmaz.

## 4. ML tuzakları (otomatik kontrol edilecekler)

1. **Bozulma sadece train'e uygulanır.** Test seti her zaman temiz kalır.
2. **Bozulma seed'e bağlı ve sabittir.** Aynı seed iki kez çalışırsa aynı permütasyon çıkar. Etiket bozulması epoch başına yeniden üretilmez, bir kez yapılır.
3. **Train accuracy, bozulmuş etiketlere göre ve augmentation'sız ölçülür.** Aksi halde ezberi yanlış ölçeriz.
4. **E1, E2 ve E4'te augmentation, dropout, weight decay kapalı.** Sadece E3'te bunlar sistematik açılır/kapanır.
5. **Yakınsama süresi karşılaştırması için lr ve schedule tüm koşullarda aynı.** Aksi halde "rastgele etiket daha yavaş" bulgusu schedule'dan gelir.
6. **Class imbalance:** CIFAR-10 dengeli, ama rastgele etiket üretiminin de sınıfları dengeli dağıttığını bir assert ile doğrularız.
7. **Validation ayrımı:** hiperparametre ayarı yaparsak test setinde yapmayız. Train'den ayrılmış validation kullanılır. Makale zaten test setini sadece raporlamak için kullanıyor.

## 5. Dizin yapısı (en küçük hali)

```
project/
  ARCHITECTURE.md      bu dosya
  README.md            nasıl çalıştırılır
  src/
    data.py            CIFAR yükleme + bozucular (etiket, piksel, gürültü)
    models.py          CNN ve MLP
    train.py           tek bir koşu: config alır, CSV üretir, checkpoint yazar
    constructive.py    E4 (uzantı)
  notebooks/           her deney için bir notebook; sadece src/'yi çağırır
  results/             koşuların CSV/JSON çıktıları
  tests/               bozucuların ve yapıcı çözümün testleri
  report/              LaTeX (kurs şablonu)
```

Notebook'lar mantık içermez, sadece `src/` fonksiyonlarını çağırır ve şekil çizer. Böylece lokalde ve Colab/Kaggle'da aynı kod çalışır.

## 6. İş bölümü

Sahiplik ayrı, anlama ortak: sunumda bireysel soru geliyor, üç kişi de **tüm projeyi ve türetimleri** açıklayabilmeli.

| Kişi | Sahip olduğu | Neden böyle |
|---|---|---|
| **A** | `data.py`, `models.py`, `train.py` (ortak kütüphane) + E1 | Diğer herkes buna bağımlı. Önce bu çıkmalı. |
| **B** | E2 (bozulma süpürmesi) + test/train gap analizi | A'nın kütüphanesi üstüne kurulur, kendi başına bir deney. |
| **C** | E3 (regularizer tablosu) + Theorem 1 türetimi + E4 (kapılı) + LaTeX rapor iskeleti | Teori ve yazım işi tek elde toplanır, ama herkes okur. |

Kurallar:
- A kütüphanenin arayüzünü (config alanları, CSV formatı) ilk iş olarak yazıp B ve C'ye verir. Sonradan değiştirilirse herkese haber verilir.
- Herkes kendi deneyinin sonuçlarını diğer ikisine anlatır. Slayt öncesi karşılıklı "bana türet" turu yapılır.
- Rapor bölümleri paylaşılır, ama LaTeX'i tek kişi birleştirir (çakışma olmasın diye).

## 7. Kapılı uzantı: Theorem 1 yapıcı ispatı (E4)

Çekirdek deneyler (E1–E3) 3 seed ile bitmeden başlanmaz. Çekirdek gecikirse düşürülür.

Yapı: rastgele `a ∈ R^d`, `z_i = a·x_i` sıralı; `b_1 < z_1 < b_2 < z_2 < … < b_n < z_n` olacak şekilde bias'lar; `A_ij = max(z_i − b_j, 0)` alt üçgenseldir ve tersinirdir; `A w = y` çözülür. Ağ `c(x) = Σ_j w_j · max(a·x − b_j, 0)`, parametre sayısı `2n + d`.

Kontroller:
- `z_i` hepsi farklı olmalı (assert). Aynı girdi iki farklı etikete sahipse kurulum imkânsız.
- `A` kötü koşullanabilir: float64 kullan, kalan hatayı (residual) raporla.
- Çözülen ağın test doğruluğunu da ölç. Beklenti: rastgele düzeyde. Bu "kapasite ≠ genelleme" noktasını gösterir.
- İsteğe bağlı: aynı boyutlu ağı SGD ile eğit, ağırlık normlarını karşılaştır (makalenin Bölüm 5'ine bağlanır).

## 8. Teslim kontrol listesi

Aşağıdakilerden biri eksikse proje **0 puan**:
- [ ] Kendi public GitHub repomuz var ve reproduction kodunu içeriyor.
- [ ] Rapor **kurs LaTeX şablonuyla** yazıldı.
- [ ] Rapor **PDF** olarak teslim edildi.

Ayrıca:
- [ ] Repo linki raporda var.
- [ ] README çalıştırma talimatını içeriyor.
- [ ] Notebook'lar Colab/Kaggle'da baştan sona çalıştı ve rapordaki sayı/şekilleri üretti.
- [ ] Her ölçek küçültmesi raporda belirtildi.
- [ ] Her konfigürasyon çoklu seed ile, mean ± std olarak raporlandı.
- [ ] Sunum: 8–10 slayt, makale özeti çok kısa, ağırlık reproduction sonuçlarında. Okuma yok.

## 9. Açık kalan şeyler

- Rapor uzunluğu ve puan dağılımı hoca tarafından henüz açıklanmadı. Açıklanınca bu dosya güncellenir.
- Kurs LaTeX şablonu (`CMP4509_report_template.zip`) Teams'ten indirilip `report/` altına konacak.

## 10. Platform uyumluluğu (Mac / Windows / Colab / Kaggle)

Ekipte farklı işletim sistemleri olabilir; Colab/Kaggle Linux'tur. Kurallar:

1. **float64 MPS'te yok.** E4 (`constructive.py`) cihazdan bağımsız her zaman `device="cpu"`, `dtype=torch.float64` ile koşar.
2. **Yollar:** `pathlib.Path` kullan, string birleştirme yok. Dosya açarken `encoding="utf-8"`.
3. **Torch kurulumu platforma özgüdür.** `requirements.txt`'e torch yazılmaz. README, pytorch.org'dan kendi platformunun komutunu almayı söyler. Windows'ta NVIDIA GPU varsa CUDA'lı wheel kurulmalı, yoksa sessizce CPU'ya düşer. `requirements.txt` sadece platformdan bağımsız paketleri (numpy, pandas, matplotlib vb.) sabitler.
4. **Sonuçlar cihazlar arasında bire bir aynı çıkmaz** (MPS/CUDA/CPU farklı kernel). Bir konfigürasyonun tüm seed'leri aynı cihazda koşulur. Raporda hangi sonucun hangi platformda üretildiği yazılır.
5. **MPS eksik operasyonlar:** README'de `PYTORCH_ENABLE_MPS_FALLBACK=1` belirtilir.
6. **Satır sonları:** repo kökünde `.gitattributes` (`* text=auto`).
7. **Smoke test:** README'de tek komutla 1 epoch + küçük alt küme koşusu bulunur (ör. `python -m src.train --config smoke`). Kütüphane ilk çıktığında B ve C bunu **kendi makinesinde** çalıştırır; uyumsuzluk ilk gün yakalanır.
