# VideoSummarizer
하이라이트 추출 모델을 학습/검증/테스트하는 기능을 구현

## 1. Data Converter
### Requirements
```
# pip install -r requirements.txt
```
or
```
# conda install --file requirements.txt
```
moviepy는 반드시 v1.0.3 버전을 설치해야 한다. Anaconda 환경에서 v1.0.3 버전이 설치되지 않을 경우에는 아래와 같이 소스를 직접 빌드하여 설치한다.
```
# git clone https://github.com/Zulko/moviepy.git
# cd moviepy
# git checkout v1.0.3
# python setup.py install
```

### Usage
```
# python data_converter.py input_video_dir segment_length(sec) video_sample_rate(fps) video_width(px) video_height(px) audio_sample_rate(hz) [output_dataset_dir]
```
Examples:
```
# python data_converter.py data/raw 5 3 64 64 11025
```
data/raw 디렉터리의 mp4, txt 파일로부터 segment 파일(.pkl)을 생성한다. 저장 디렉터리는 현재 디렉터리 하위에 생성된다.<br>
이 때, segment 길이는 5초, 비디오 프레임에서 초당 3프레임을 추출하며 해상도는 64x64픽셀이 된다. 오디오는 11025Hz로 변환하여 추출한다.
```
# python data_converter.py data/raw 3 6 128 128 22050 data/dataset1
```
data/raw 디렉터리의 mp4, txt 파일로부터 segment 파일(.pkl)을 생성한다. 변환된 파일은 data/dataset1에 저장된다.<br>
이 때, segment 길이는 3초, 비디오 프레임에서 초당 6프레임을 추출하며 해상도는 128x128픽셀이 된다. 오디오는 22050Hz로 변환하여 추출한다.
