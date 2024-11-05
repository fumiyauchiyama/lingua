FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel

# 作業ディレクトリの設定
WORKDIR /workspace

# 必要なパッケージのインストールとsudoers.dディレクトリの作成
RUN apt-get update && \
    apt-get install -y sudo git && \
    mkdir -p /etc/sudoers.d && \
    rm -rf /var/lib/apt/lists/*

# 新しいユーザーの作成とsudo権限の設定
RUN useradd -m -s /bin/bash developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer

# 必要なPythonパッケージのインストール
COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121 && \
    pip install ninja && \
    pip install -r requirements.txt

# デフォルトのユーザーを 'developer' に設定
USER developer

# デフォルトの作業ディレクトリをユーザーのホームディレクトリに設定
WORKDIR /home/developer

# デフォルトのコマンド（必要に応じて変更）
CMD ["bash"]
