#build.sh
docker build -t mj-test-cu121 \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -f Dockerfile .