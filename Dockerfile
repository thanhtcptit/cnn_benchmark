FROM tensorflow/tensorflow:1.13.2-gpu-py3

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev

COPY requirements.txt /project/

RUN pip3 install -r /project/requirements.txt

COPY ml_best_practice /project/ml_best_practice

RUN pip3 install -e /project/ml_best_practice/nsds

COPY cnn_benchmark /project/cnn_benchmark

COPY run.py /project/

WORKDIR /project/

CMD adduser -u $CURRENT_UID host_user --no-create-home --disabled-password --gecos "" && \
    su host_user -c "python3 run.py benchmark run_configs/benchmark.json --show_logs=$SHOW_LOGS"
