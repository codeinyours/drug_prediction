import numpy as np
from pandas import DataFrame
from rdkit.Chem import AllChem
from sklearn import metrics
from tabulate import tabulate


def get_data(df: DataFrame):
    """
    :param df: smiles가 포함된 데이터프레임
    1. smiles 컬럼에 MolFromSmiles를 통해 Molecule 정보로 변환합니다.
    2. GetMorganFingerprintAsBitVect를 통해 이진 벡터로 변환합니다.
    2. 훈련용 데이터와 검증용 데이터로 나누어 반환합니다.
    :return:
        input: 훈련용 데이터
        target: 검증용 데이터
    """
    input = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=2048) for x in df['molecule']])
    target = df['activity'].eq('active').astype(float).to_numpy()

    return input, target

    
def show_metric(y_true, y_pred):
    logloss = metrics.log_loss(y_true, y_pred, labels=[0, 1])

    # AUC PRC
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    auc_prc = metrics.auc(recall, precision)

    # AUC ROC
    fpr_roc, tpr_roc, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc_roc = metrics.auc(fpr_roc, tpr_roc)

    print(tabulate([['LOG LOSS', logloss], ['AUC PRC', auc_prc], ['AUC ROC', auc_roc]], headers=['METRIC', 'SCORE']))