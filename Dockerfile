# syntax=docker/dockerfile:1
FROM ubuntu:22.04
LABEL maintainer="cb7454@princeton.edu"

COPY ./requirements.txt /requirements.txt

WORKDIR /docker_thermo_ai
ENV PYTHONPATH=/docker_thermo_ai:/docker_thermo_ai/projects

# ubuntu dependencies
RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    build-essential \
    python3-pip

RUN mkdir -p $VSCODE_INSTALL_DIR/data
RUN curl -L $VSCODE_BINARY_URL -o /tmp/vscode-linux-x64.tar.gz \
    && tar -zxvf /tmp/vscode-linux-x64.tar.gz --directory $VSCODE_INSTALL_DIR --strip-components=1
ENV PATH=$VSCODE_INSTALL_DIR/bin:$PATH


# install python dependencies
RUN pip install -r /requirements.txt \
&& rm -f /requirements.txt

CMD ["/bin/bash"]