# TensorFlow Example

## 오승빈

### AVX 에러 메세지 제거

출력창에서 결과 출력 전 경고가 떴을 때

> Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

코드를 추가하여 해결

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
