FROM python:3.9.10-buster
USER root

RUN apt-get update && apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

#RUN apt install -y libopencv-dev
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /root/
RUN  pip install -r /root/requirements.txt
