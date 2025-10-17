#run.sh --rm 제거
docker run -it --gpus all \
    --name mj-test \
    --shm-size=8g \
    -v /home/mj/skeleton-based-action-recognition:/workspace \
    -w /workspace \
    mj-test-cu121 \