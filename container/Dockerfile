# syntax=docker/dockerfile:1
FROM ubuntu:22.04
LABEL maintainer="cb7454@princeton.edu"

ENV VSCODE_INSTALL_DIR=/opt/VSCode-linux-x64
ENV VSCODE_BINARY_URL="https://code.visualstudio.com/sha/download?build=stable&os=linux-x64"
COPY ./requirements.txt /requirements.txt

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/workspace/projects

# ubuntu dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    build-essential \
    curl \
    python3-pip \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    gmsh \
    zsh \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcups2 \
    libxss1 \
    libxshmfence1 \
    libgbm1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p $VSCODE_INSTALL_DIR/data
RUN curl -L $VSCODE_BINARY_URL -o /tmp/vscode-linux-x64.tar.gz \
    && tar -zxvf /tmp/vscode-linux-x64.tar.gz --directory $VSCODE_INSTALL_DIR --strip-components=1
ENV PATH=$VSCODE_INSTALL_DIR/bin:$PATH


# install python dependencies
RUN pip install -r /requirements.txt \
&& rm -f /requirements.txt

CMD ["/bin/bash"]