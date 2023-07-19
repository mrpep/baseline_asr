import joblib 
import tqdm
import os 
from pathlib import Path

data_path='/mnt/matylda5/qpepino/baseline-asr/chime_submit/mixer6_deveval.pkl'
output_path='/mnt/matylda5/qpepino/baseline-asr/chime_submit/mixer6/deveval'

data=joblib.load(data_path)
n_best=20

for n in tqdm.tqdm(range(n_best)):
    path=str(n+1)+'best_recog'
    dest_path = Path(output_path,path)
    if not dest_path.exists():
        dest_path.mkdir(parents=True)

    with open(Path(dest_path,'text'),'w') as file:
        for i in tqdm.tqdm(range(len(data['ids']))):
            file.write(data['ids'][i]+' '+data['nbest'][i][n]+'\n')
    
    with open (Path(dest_path,'score'),'w') as file:
        for i in tqdm.tqdm(range(len(data['ids']))):                
            file.write(data['ids'][i]+' '+str(data['nbest_scores'][i][n])+'\n')
        
           

       
            
        
            