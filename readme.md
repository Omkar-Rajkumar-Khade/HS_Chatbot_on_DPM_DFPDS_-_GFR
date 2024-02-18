# Chatbot on DPM, DFPDS & GFR
This repository contains the codebase for an Intelligent Chatbot System.  Our primary goal for this system is to use Generative AI technologies to create a robust chatbot capable of providing intelligent responses based on DPM, DFPDS & GFR Documents. The system is specifically designed to provide intelligent responses based on DPM (DEFENCE PROCUREMENT MANUAL, 2009 ), DFPDS (DELEGATION OF FINANCIAL POWERS TO DEFENCE SERVICES- 2021), and GPR (GENERAL FINANCIAL RULES 2017) Documents. It employs various components and technologies to achieve this functionality.


## Application Setup

* step-1: Visit Url: https://github.com/Omkar-Rajkumar-Khade/HS_Chatbot_on_DPM_DFPDS_-_GFR/releases/tag/v1.0.0
* step-2: Download `hsbot.exe` and `setup.exe`
* step-3: Execute setup.exe
* setp-4: Execute hsbot.exe 


## `Troubleshooting Database`

##### To start VectorDB container
###### For Linux
```bash
sudo docker-compose up -d
```

###### For Windows
```bash
docker-compose up -d
```

##### To Stop VectorDB container
###### For Linux
```bash
sudo docker-compose down
```

###### For Windows
```bash
docker-compose down
```

## Repository Structure
`.github/workflows`
  - main.yml (GitHub Actions workflow for automated CI/CD)
  
`data`
  - (Directory containing the dataset or relevant data)

`elasticsearch`
  - (Elastic Search related configurations and files)

`Chatbot on DFPDS.ipynb`
  - (Jupyter Notebook for developing the chatbot specifically for DFPDS)

`Chatbot on DPM, DFPDS & GRF.ipynb`
  - (Jupyter Notebook for developing the chatbot covering DPM, DFPDS, and GRF)

`Dockerfile`
  - (Docker configuration file for containerized deployment)

`app.py`
  - (Main application script)

`hsbot.exe`
  - (Executable file for standalone deployment)

`readme.md`
  - (This readme file)

`requirements.txt`
  - (List of Python dependencies)

`run.ps1`
  - (PowerShell script for running the application)

`setup.exe`
  - (Executable file for installation/setup)

`setup.ps1`
  - (PowerShell script for setup)

## Tech Stack

1. Embedding Generation: Utilizes `GTE (Generic Text Embeddings)` embeddings for text representation.
2. VectorDB: `Elastic Search` is utilized as VectorDB for efficient similarity search and retrieval.
3. LLM (Language and Learning Model): `Mistral 7B` (LLM) is employed for language understanding and generation tasks.
4. Framework: The project is built upon the `Haystack` framework for streamlined development and integration of various NLP components.
5. Frontend: `Streamlit` is used for the user interface, providing an interactive experience for users.
6. Model Deployment: Models are deployed either using `Docker` containers or as standalone `executables (Exe)`.
