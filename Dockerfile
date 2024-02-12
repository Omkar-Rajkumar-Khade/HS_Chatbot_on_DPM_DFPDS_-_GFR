FROM nvcr.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
WORKDIR /app
COPY  . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT streamlit run app.py
