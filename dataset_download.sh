mkdir data/
cd data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BqxIklysgCJXq1SJsX-Jlw98e2mg56wf' -O- | sed -rm 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BqxIklysgCJXq1SJsX-Jlw98e2mg56wf" -O train_data.csv && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h8agdS6_3DLGEzrue7rsxR7Z8kr4A8Gz' -O- | sed -rm 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h8agdS6_3DLGEzrue7rsxR7Z8kr4A8Gz" -O test_data.csv && rm -rf /tmp/cookies.txt

echo Dataset has been downloaded in data/