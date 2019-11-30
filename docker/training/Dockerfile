FROM tensorflow/tensorflow:1.10.0-gpu-py3

WORKDIR /cnn_benchmark/

RUN pip install -r requirements.txt

CMD python run.py train run_configs/train.json
