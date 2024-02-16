$repoUrl = "https://github.com/Omkar-Rajkumar-Khade/HS_Chatbot_on_DPM_DFPDS_-_GFR"
$targetDir = "$env:USERPROFILE\HS_Chatbot_on_DPM_DFPDS_-_GFR"
git clone $repoUrl $targetDir
cd $targetDir
pip install -r requirements.txt
cd elasticsearch
docker-compose up -d 
Write-Output "Setup Complete"
