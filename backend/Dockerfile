# backend/Dockerfile

FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8090

CMD ["python", "main.py"]
