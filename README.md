### Prerequisites

가상 환경을 활성화합니다.  
`$ source venv/bin/activate`

프로젝트에 필요한 패키지를 설치해줍니다.  
`$ pip install -r requirements.txt`

훈련에 사용하고 싶은 데이터셋을 dataset/ 폴더 아래에 준비합니다.


### 사용법

`$ python activity_prediction.py`

외부 데이터셋을 사용하고 싶은 경우, 데이터셋 url과 함께 다음과 같이 실행합니다.

`$ python activity_prediction.py --url {DATASET_DOWNLOAD_URL}`

