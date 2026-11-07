# Docker image for the v2 rerun: same steps as env/apptainer.def (both run env/install_env.sh).
#   docker build -t repattack:v2 -f env/Dockerfile .
#   docker run --gpus '"device=0"' -v $PWD:/work -w /work -v $RERUN_ROOT:/rerun -e RERUN_ROOT=/rerun repattack:v2 ./reproduce.sh status
FROM ubuntu@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip gcc g++ git curl unzip ca-certificates p7zip-full procps util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY env /opt/repattack/env
RUN bash /opt/repattack/env/install_env.sh /opt/repattack \
    && { gcc --version | head -1; g++ --version | head -1; /opt/repattack/jdk/bin/java -version 2>&1 | head -1; \
         /opt/repattack/venv/bin/python --version; } > /opt/repattack/VERSIONS \
    && /opt/repattack/venv/bin/pip freeze > /opt/repattack/pip-freeze.txt

ENV RERUN_PY=/opt/repattack/venv/bin/python RERUN_JDK=/opt/repattack/jdk \
    RERUN_JOERN=/opt/repattack/joern-v4.0.655 RERUN_UPSTREAM=/opt/repattack/upstream \
    RERUN_IMAGE=repattack-v2 PATH=/opt/repattack/venv/bin:/opt/repattack/jdk/bin:$PATH
WORKDIR /work
CMD ["./reproduce.sh", "status"]
