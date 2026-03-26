## 1. Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesi ile İlişkisi

### Temel Tanımlar
* **Matris Manipülasyonu:** Verilerin çok boyutlu diziler (matrisler) halinde ifade edilerek üzerlerinde toplama, çarpma, devrik (transpose) ve ters alma (inverse) gibi lineer cebir işlemlerinin uygulanmasıdır.
* **Özdeğer (Eigenvalue) ve Özvektör (Eigenvector):** Karesel bir $A$ matrisi, sıfırdan farklı bir $v$ vektörü ile çarpıldığında, vektörün yönünü değiştirmeden sadece büyüklüğünü $\lambda$ katsayısı kadar ölçeklendiriyorsa; $v$ vektörüne **özvektör**, $\lambda$ değerine ise **özdeğer** denir ($Av = \lambda v$).

### Makine Öğrenmesi İle İlişkisi
Makine öğrenmesi algoritmaları, yüksek boyutlu veri setlerini matris formunda temsil ederek işler. Matris manipülasyonu, bu devasa verilerin donanım üzerinde paralel ve verimli bir şekilde hesaplanmasını sağlar. Özdeğerler ve özvektörler ise veri matrisinin yapısal özelliklerini (DNA'sını) analiz etmek için kullanılır. Özvektörler, verideki değişimin (varyansın) yönlerini ifade ederken; özdeğerler bu yönlerin taşıdığı bilgi miktarını, yani makine öğrenmesi modeli için ne kadar önemli olduklarını gösterir.

### Kullanıldığı Temel Yöntemler ve Yaklaşımlar
* **Temel Bileşenler Analizi (PCA):** Verideki gürültüyü azaltmak ve boyut indirgemek için kullanılır. Verinin kovaryans matrisinin özdeğer ve özvektörleri hesaplanır, en büyük özdeğere sahip yönler (temel bileşenler) modele dahil edilir.
* **Tekil Değer Ayrışımı (SVD):** Karesel olmayan matrisler için de çalışan genelleştirilmiş bir ayrımdır. Özellikle doğal dil işleme (NLP) uygulamalarında ve tavsiye sistemlerinde (recommendation systems) boyut küçültme ve gizli faktörleri bulma amacıyla sıkça kullanılır.
* **Spektral Kümeleme:** Karmaşık yapıdaki veri noktalarını gruplamak için, veri noktaları arasındaki benzerlikleri temsil eden grafın (Laplacian matrisi) özdeğer ve özvektörlerinden yararlanır.

### Referanslar
* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press.

## 2. Numpy linalg.eig Fonksiyonunun İncelenmesi

Bu bölümde, Python'ın bilimsel hesaplama kütüphanesi Numpy'ın `linalg` (lineer cebir) modülü altında bulunan `eig` fonksiyonunun dokümantasyonu ve arka plandaki çalışma mantığı incelenmiştir.

### Fonksiyonun Amacı
`numpy.linalg.eig(a)` fonksiyonu, karesel bir `a` matrisinin özdeğerlerini (eigenvalues) ve sağ özvektörlerini (right eigenvectors) eşzamanlı olarak hesaplamak için kullanılır.

### Parametreler ve Geri Dönüş Değerleri (Dokümantasyon)
* **Parametre (`a`):** Fonksiyon, parametre olarak nxn boyutlarında karesel bir matris (array_like) alır. Eğer matris karesel değilse `LinAlgError` hatası fırlatır.
* **Geri Dönüş (Returns):** Fonksiyon iki farklı dizi (array) döndürür: `w` ve `v`.
  * **`w` (Özdeğerler):** Matrisin özdeğerlerini içeren tek boyutlu bir dizidir. Dokümantasyonda özellikle belirtildiği üzere, bu özdeğerler belirli bir büyüklük sırasına göre **sıralanmış olmak zorunda değildir**.
  * **`v` (Özvektörler):** Sütunları özvektörleri temsil eden 2 boyutlu karesel bir matristir. `v[:, i]` sütunu, `w[i]` özdeğerine karşılık gelen normalize edilmiş özvektördür (yani vektörün normu 1'dir).

### Kaynak Kod ve Arka Plan İşleyişi (Under the Hood)
Numpy'ın kaynak kodları incelendiğinde, yüksek performanslı matematiksel işlemler için Numpy'ın kendi başına tekerleği yeniden icat etmediği görülür. `numpy.linalg.eig` fonksiyonu çağrıldığında, alt seviyede C ve Fortran dilleriyle yazılmış standart ve son derece optimize edilmiş **LAPACK (Linear Algebra PACKage)** kütüphanesine başvurulur.

**Adım Adım Arka Plan İşlemleri:**
1. **LAPACK Rutinlerinin Çağrılması:** `eig` fonksiyonu, veri tipine (float, double, complex) bağlı olarak LAPACK'ın `_geev` (örneğin çift hassasiyetli reel sayılar için `DGEEV`) rutinini çağırır.
2. **Hessenberg İndirgemesi:** LAPACK, doğrudan hesaplama yapmak yerine önce girdi matrisini daha kolay işlenebilir bir form olan "Upper Hessenberg" formuna dönüştürür.
3. **QR Algoritması:** Hessenberg formundaki matrise iteratif olarak QR algoritması uygulanarak özdeğerler ve özvektörler yüksek hassasiyetle hesaplanır.

*Önemli Not:* Dokümantasyonda yer alan önemli bir uyarıya göre; eğer üzerinde çalışılan matris simetrik (reel simetrik veya kompleks Hermitian) bir matris ise, genel amaçlı `eig` fonksiyonu yerine simetrik matrisler için özel olarak optimize edilmiş `numpy.linalg.eigh` fonksiyonunun kullanılması hem daha hızlı sonuç verir hem de sayısal olarak daha kararlıdır.
