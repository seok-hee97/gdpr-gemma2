# GPU 지원을 위한 CUDA 기반 이미지 사용
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir rouge_score evaluate tqdm pandas

# 프로젝트 코드 복사
COPY . .

# 모델 폴더는 용량이 크므로 볼륨 마운트를 권장하지만, 
# 필요한 경우 COPY gdpr_gemma_2b ./gdpr_gemma_2b 를 수행할 수 있습니다.

# 실행 권한 부여
RUN chmod +x src/*.py

CMD ["python3", "-m", "src.eval"]
