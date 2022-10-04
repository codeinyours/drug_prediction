import os.path

import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier

from utils import get_data, show_metric, bcolors, get_option


if not os.path.exists(DEFAULT_CSV_PATH) and not csv_url:
    raise FileNotFoundError(f'dataset 디렉토리 아래에 cmpd.csv가 존재하지 않습니다.\n\t'
                            f'1. {bcolors.BOLD}dataset 디렉토리에 데이터셋이 존재하는지\n\t'
                            f'2. 파일명에 문제가 있는지 확인해주세요.{bcolors.ENDC}')

df = pd.read_csv(csv_url if csv_url else DEFAULT_CSV_PATH)
df['molecule'] = df['smiles'].apply(Chem.MolFromSmiles)

train_input, train_target = get_data(df.loc[df['group'] == 'train'])
test_input, test_target = get_data(df.loc[df['group'] == 'test'])

rf = RandomForestClassifier()
rf.fit(train_input, train_target)
rf.score(test_input, test_target)

y_pred = rf.predict_proba(test_input)[:, 1]

show_metric(y_true=test_target, y_pred=y_pred)
