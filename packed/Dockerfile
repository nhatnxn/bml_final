# FROM python:3.6 AS compile-image
# COPY . ./
# RUN apt-get update
# RUN pip3 install --no-cache-dir --user -r requirements.txt
# RUN apt-get install -y libgl1-mesa-dev
# EXPOSE 8000
# CMD python3 manage.py makemigrations
# CMD python3 manage.py migrate
# CMD python3 manage.py runserver 0.0.0.0:8000

FROM ubuntu
#FROM python:3.6 AS compile-image
#FROM alangobrown/python-rabbit-consumer
#RUN add-apt-repository ppa:jonathonf/python-3.6
#RUN apt-get update
#RUN echo "deb http://archive.ubuntu.com/ubuntu/
RUN apt-get update
RUN apt-get install -y tar git curl nano wget dialog net-tools build-essential
#RUN wget https://www.python.org/ftp/python/3.4.5/Python-3.4.5.tgz
#RUN tar zxfv Python-3.4.5.tgz
#RUN find ~/filename/python -type d | xargs chmod 0755
#RUN cd Python-3.4.5/
#RUN ./configure --prefix=$HOME/filename/python
#RUN make
#RUN make install
#RUN apt-get install -y python3 python3-dev python3-pip
RUN apt-get install -y python3.6
RUN apt-get update
RUN apt-get install -y python3-pip
#RUN apt-get install -y rabbitmq-server

COPY . ./
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt
# RUN apt-get install -y libgl1-mesa-dev

#RUN mkdir -p /etc/rabbitmq /var/lib/rabbitmq /var/log/rabbitmq && \
#    chown rabbitmq:rabbitmq /etc/rabbitmq /var/lib/rabbitmq /var/log/rabbitmq && \
#    ln -s /usr/lib/rabbitmq-server/sbin/cuttlefish /usr/bin/

#CMD sudo rabbitmqctl add_user test testpass1
#CMD sudo rabbitmqctl add_vhost myvhost
#CMD sudo rabbitmqctl set_permissions -p myvhost test ".*" ".*" ".*"
#CMD sudo rabbitmq-server
#RUN service rabbitmq-server start

COPY . /code/
WORKDIR /code/
EXPOSE 8501
CMD streamlit run demo.py
#CMD celery -A Data_Backup worker -l info -B
#CMD ["rabbitmq-server"]
#CMD service rabbitmq-server start && celery -A Data_Backup worker -l info -B
#CMD celery worker --app=Backup.tasks
#CMD brew services start rabbitmq