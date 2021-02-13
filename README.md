# mil-audio-denoising

## Тестовое задание

Это небольшой отчет о ходе выполнения тестового задания. Нужно было реализовать простую сеть для audio denoising. 

### Вступление
Есть методы устранения шума в изображниях или в аудиосигналах с помощью отображения исходного изображения/аудио в другое пространство, в котором эти "данные" имеют разреженное представление. Например, если сделать 2D преобразования Фурье для изображения (преобразование Фурье для аудио), то окажется, что очень многие коэффициенты такого отображения очень близки к нулю, а, значит, и вклад в "значимую состовляющую данных" этих коэффициентов тоже невелик. Если занулить такие коэффициенты, а потом сделать обратное преобразование Фурье, то полученное изображение или аудио почти не потеряют в качестве, или могут даже в качестве бонуса избавиться от шумовых составляющих.

В основе ряда методов denoising'а лежит сложное нелиенйное преобразование в другое пространство (латентное представление), а затем обратное преобразование переводит латентное представление в "исходные координаты", и получается исходный аудиосигнал с подавленными шумовыми составляющими.

#### Зашумление данных

<details>
  <summary><strong>Ход решения задания</strong></summary>
    
#### 1. Простейший автоенкодер
Я решил начать с совсем простого автоенкодера, с одним сверточным слоем енкодера и одним сверточным слоем декодера, как в этом [примере](https://github.com/GuitarsAI/MLfAS/blob/master/MLAS_07_Denoising_Autoencoder.ipynb). Автор показывает на примере одной песни, что шумы хоть и остаются, но становится их заметно меньше.

```
Encoder = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2048, stride=32, padding=1023, bias=True)
Decoder = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2048, stride=32, padding=1023, bias=True)
```

- Из датасета я взял небольшую часть записей для проверки того, будет ли происходить хотя бы качественный overfitting, и будет ли модель хотя бы делать тождественное преобразование, восстанавливая исходный сигнал при подаче на вход исходного сигнала.
- Проверка показала, что модель не в состоянии делать тождественное преобразование. Видимо, у автора примера произошел overfitting на одной единственной песне.

После того, как самая простая модель не удалась, я продолжил искать существующие решения. Второе что я попробовал - это Denoising WaveNet.
#### 2. Denoising Wavenet
* [arxiv.org A Wavenet for Speech Denoising](https://arxiv.org/abs/1706.07162)
* [github](https://github.com/saurav-pathak/WaveNet_PyTorch)

<img align="left" src = "pics/wavenet.png" width ="400" /> <img src = "pics/wavenet2.png" width ="300" />


С нуля эту модель я не реализовывал, но пришлось ее немного поменять, убрал ненужные элементы, связанные с идентификацией говорящего. Подстроил под уже написанный код. Исходная конфигурация, описанная в github, получилась довольно тяжелой для моих мощностей (NVIDIA GeForce GTX 1080 Ti).

Чтобы обрабатывать записи мини-батчами, пришлось делать их одной длины, и короткие записи становятся длиной с максимальную запись в мини-батче. У модели нет фиксированного размера летентного представления (так как по сути всё строится на сверточных слоях, без Fully Connected слоев посередине), поэтому нет ограничения на длину входящих записей и они могут обрабатываться целиком. Но при вычислениях градиентов (даже мини-батч размера 2) на видеокарте не хватало памяти. Тогда я подсмотрел решение в [Wave-U-Net](https://github.com/f90/Wave-U-Net-Pytorch), где входящий мини-батч обрабатывается кусочками, которые последовательно подаются на вход сети.

- Так же как и с первой моделью, для начала решил проверить, насколько модель способна выучить тождественное преобразование на небольшом датасете. Судя по тому, что у сети есть skip-connections, у нее должно легко это получиться.
- Так и получилось. Но, к сожалению, сеть даже на небольшом датасете обучалась у меня довольно долго. При уменьшении числа Residual слоев и степени dilation модель обучалась несколько быстрее.
- А вот восстанавливать исходный сигнал из зашумленого у модели не получилось (тут нужно заметить, что на этот момент я только занулял коэффициенты STFT-представления, другие виды зашумления не делал). Из метрик здесь пробовал L1 и MSE.
- Так как модель обучалась медленно и не показала хорошего результата, я решил поискать что можно попробовать еще.
####  3. Deep Complex U-Net
* [arxiv.org PHASE-AWARE SPEECH ENHANCEMENT WITH DEEP COMPLEX U-NET](https://openreview.net/pdf?id=SkeRTsAcYm)
* [github](https://github.com/pheepa/DCUnet)
<img src = "pics/dcunet.png" width ="800" /> 
    
Так же с нуля модель не реализовывал, переделал, чтобы на вход сеть принимала waveform, а не STFT-представление, убрал лишние преобразования, связанные с этим, больше экспериментировал с накладыванием шумов.
- Сначала так же проверил, что модель способна делать тождественное преобразование.
- В качестве метрики используется wSDR - weightted source to distortion ratio - с помощью которой оценивал качество зашумленных версий сигналов по сравнению с чистыми и качество "очещенных" сетью.
- Опять же, в начале пробовал только обнуление коэффициентов STFT-представления. Судя по метрике, качество сигнала после обработки сетью улучшалось, но не сильно. На слух так вообще не заметно. Но, возможно, модель не очень хорошо справляется только с таким видом зашумлений - речь становится искаженной.
- Затем я попробовал наоборот усиливать маленькие коэффициенты STFT-представления - поднимать их до 1.
    
    
</details>