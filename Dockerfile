FROM python:3.11
WORKDIR /app
COPY  . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT streamlit run app.py