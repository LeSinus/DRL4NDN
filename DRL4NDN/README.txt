conda create --name DRL -c pytorch -c conda-forge --file conda-requirements.txt

conda activate DRL

pip install tqdm==4.66.1 stable-baselines3==2.2.1 sb3-contrib==2.2.1 rich==13.7.0 pytz==2023.3.post1 pygments==2.17.2 mdurl==0.1.2 markdown-it-py==3.0.0 fsspec==2023.12.0
