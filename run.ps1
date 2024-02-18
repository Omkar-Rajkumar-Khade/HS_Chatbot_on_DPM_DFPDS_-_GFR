$targetDir = "$env:USERPROFILE\HS_Chatbot_on_DPM_DFPDS_-_GFR"
cd $targetDir
cd elasticsearch
docker-compose up -d 
cd ..
streamlit.exe run app.py
