# syntax=docker/dockerfile:1

FROM ubuntu:22.04
LABEL maintainer="cb7454@princeton.edu"

COPY ./requirements.txt /requirements.txt

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/workspace/projects

# ubuntu dependencies
RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    build-essential \
    python3-pip

# install python dependencies
RUN pip install -r /requirements.txt \
&& rm -f /requirements.txt

CMD ["/bin/bash"]