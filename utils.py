import argparse
from typing import Optional
import numpy as np
import pandas as pd
from pandas import DataFrame
from rdkit.Chem import AllChem
from sklearn import metrics
from tabulate import tabulate
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


DEFAULT_CSV_PATH = "dataset/cmpd.csv"


def unescaped_str(url: str) -> str:
    return url.replace('\\', '')


def get_df() -> pd.DataFrame:
    DEFAULT_CSV_PATH = "dataset/cmpd.csv"
    csv_url = get_option()

    if not os.path.exists(DEFAULT_CSV_PATH) and not csv_url:
        raise FileNotFoundError(f'dataset 디렉토리 아래에 cmpd.csv가 존재하지 않습니다.\n\t'
                                f'1. {bcolors.BOLD}dataset 디렉토리에 데이터셋이 존재하는지\n\t'
                                f'2. 파일명에 문제가 있는지 확인해주세요.{bcolors.ENDC}')

    df = pd.read_csv(csv_url if csv_url else DEFAULT_CSV_PATH)

    return df


def get_option() -> Optional[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',
                        required=False,
                        default=None,
                        type=unescaped_str,
                        help='compound dataset을 다운로드할 수 있는 URL 주소'
                        )

    args = parser.parse_args()
    return args.url


def get_data(df: DataFrame):
    """
    :param df: smiles가 포함된 데이터프레임
    1. smiles 컬럼에 MolFromSmiles를 통해 Molecule 정보로 변환합니다.
    2. GetMorganFingerprintAsBitVect를 통해 이진 벡터로 변환합니다.
    2. 훈련용 데이터와 검증용 데이터(Y)로 나누어 반환합니다.
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