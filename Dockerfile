FROM nvcr.io/nvidia/pytorch:23.12-py3
WORKDIR /app
COPY  . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT streamlit run app.py