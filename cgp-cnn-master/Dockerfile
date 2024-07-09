FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# メンテナラベル
LABEL maintainer="yabukazuki <yabuuchi@ss.cs.osakafu-u.ac.jp>"

# コンテナの作業場所の決定
ENV WORKSPACE /var/www
WORKDIR $WORKSPACE

# Poerty のインストール
RUN pip install --upgrade pip && \
          pip install poetry
ENV PATH /root/.poetry/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive

# # (初めて環境構築する際は, ここはコメントアウト)
COPY pyproject.toml $WORKSPACE
COPY poetry.lock $WORKSPACE
COPY requirements.txt $WORKSPACE

# Poetry でインストールできないライブラリをインストール
RUN pip install --ignore-installed -r requirements.txt

# Poetry によるライブラリの依存関係の解決 (初めて環境構築する際は, ここはコメントアウト)
RUN poetry config virtualenvs.create false && \
    pip install --upgrade pip && \
    pip install -U setuptools && \
    poetry install --no-root

# 8. エントリーポイントの指定
CMD ["apt-get", "update"]