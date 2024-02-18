$repoUrl = "https://github.com/Omkar-Rajkumar-Khade/HS_Chatbot_on_DPM_DFPDS_-_GFR"
$targetDir = "$env:USERPROFILE\HS_Chatbot_on_DPM_DFPDS_-_GFR"
git clone $repoUrl $targetDir
cd $targetDir
pip install -r requirements.txt
Write-Output "Setup Complete"
