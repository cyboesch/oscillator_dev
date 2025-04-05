# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/jax:24.04-py3
LABEL maintainer="cb7454@princeton.edu"

ENV USER=cb7454
ENV VSCODE_INSTALL_DIR=/opt/VSCode-linux-x64
ENV VSCODE_BINARY_URL="https://code.visualstudio.com/sha/download?build=stable&os=linux-x64"

WORKDIR /root

RUN apt-get update && apt-get install -y \
    neofetch \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    gmsh \
    zsh \
    # VS Code GUI deps
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

SHELL ["/bin/zsh", "-c"]


COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
RUN pip install jupyterlab ipykernel

RUN mkdir -p $VSCODE_INSTALL_DIR/data
RUN curl -L $VSCODE_BINARY_URL -o /tmp/vscode-linux-x64.tar.gz \
  && tar -zxvf /tmp/vscode-linux-x64.tar.gz --directory $VSCODE_INSTALL_DIR --strip-components=1
ENV PATH=$VSCODE_INSTALL_DIR/bin:$PATH

RUN addgroup --gid 1000 $USER \
 && adduser --uid 1000 --ingroup $USER --home /home/$USER --shell /usr/bin/zsh --disabled-password --gecos "" $USER \
 && chown -R $USER:$USER /home/$USER \
 && chown -R $USER:$USER $VSCODE_INSTALL_DIR

RUN curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | \
    tar -C /usr/local/bin -xzf - \
 && chown root:root /usr/local/bin/fixuid \
 && chmod 4755 /usr/local/bin/fixuid \
 && mkdir -p /etc/fixuid \
 && echo -e "user: $USER\ngroup: $USER" > /etc/fixuid/config.yml

# Fix permissions while still root
RUN chown -R $USER:$USER /home/$USER

USER $USER:$GROUP
ENV HOME=/home/$USER
WORKDIR /home/$USER
