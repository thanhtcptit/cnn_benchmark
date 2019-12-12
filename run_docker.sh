
#!/bin/bash
nvidia-docker run --rm --name cnn_benchmark \
-e CURRENT_UID=$(id -u) \
-e SHOW_LOGS=0 \
--mount type=bind,source="$(pwd)"/run_configs,target=/project/run_configs \
--mount type=bind,source="$(pwd)"/data/,target=/project/data \
--mount type=bind,source="$(pwd)"/train_logs/,target=/project/train_logs \
cnn_benchmark:dev