from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier

from utils import get_data, show_metric, get_df


df = get_df()
df['molecule'] = df['smiles'].apply(Chem.MolFromSmiles)

train_input, train_target = get_data(df.loc[df['group'] == 'train'])
test_input, test_target = get_data(df.loc[df['group'] == 'test'])

rf = RandomForestClassifier()
rf.fit(train_input, train_target)
rf.score(test_input, test_target)

y_pred = rf.predict_proba(test_input)[:, 1]

show_metric(y_true=test_target, y_pred=y_pred)
