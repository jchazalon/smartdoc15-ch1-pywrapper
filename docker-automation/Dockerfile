# Use the latest Ubuntu image as the base
FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /app

# Install Python 2.7 and pip
RUN apt-get update

RUN apt-get install -y python2.7 curl
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
RUN python2.7 get-pip.py
RUN pip install --upgrade pip
RUN apt-get install -y build-essential
RUN apt-get install -y python2.7-dev

COPY docker-py-requirements.txt /app/requirements.txt

# Install required Python packages
RUN pip install -r requirements.txt

# Run run.py when the container launches
CMD python2.7 setup.py build && python2.7 setup.py install && pip install smartdoc15_ch1 && python2.7 docker-automation/docker-entrypoint.py