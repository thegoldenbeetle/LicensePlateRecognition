# LicensePlateRecognition
DL course

Для настройки среды:
- `pip install -r ./pip.txt`

Для запуска пайплайна необходимо предварительно скачать веса моделей:
- [Веса](https://drive.google.com/file/d/17fULY0JcZbc80XiKCAFj2RM05NY7eTLI/view?usp=sharing) модели детектирования машин поместить в `./`
- [Веса](https://drive.google.com/file/d/13rR4LvpLMadaTwiT-I239zqhO_EiG3PL/view?usp=sharing) модели детектирования номеров поместить в `plate_detection/`
- [Веса](https://drive.google.com/file/d/1zaEFWAjQrEpa4fQbooirG4stKOJ6SATo/view?usp=sharing) модели распознавания номеров поместить в `number_recognition/`

А также клонировать официальный репозиторий yolov5:
`git clone https://github.com/ultralytics/yolov5`

Структура репозитория:
- В папке dev скрипт для проверки на pep8 `./dev/fix_codestyle.sh`

### Про распознавание номеров:
Идея [модели](https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8) распознавания номеров.

Документ с подробным описанием: https://docs.google.com/document/d/1ic4dMmIda8d0HcPVK3ijlawNiIg-R8MzXLyj9Z124i0/edit?usp=sharing
