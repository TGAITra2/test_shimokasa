"""未収予測推論を実行するスクリプト"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm
import mlflow
import pandas as pd
from common.constants import TEST_USE_COLS_TYPE
from common.logger import logger


def get_args() -> argparse.Namespace:
    """入力パラメータを取得する関数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Path of ML model used for inference")
    parser.add_argument("--model_version", type=str, default="latest", help="Version of model to load")
    parser.add_argument("--feature_path", type=str, help="Feature file path")
    parser.add_argument("--output_path", type=str, help="Path to store inference results")

    return parser.parse_args()


def read_input_feature(filepath_or_buffer: str) -> pd.DataFrame:
    """inputファイルを読み込む関数"""
    try:
        loaded_feature = pd.read_csv(filepath_or_buffer, encoding="utf-8", dtype=TEST_USE_COLS_TYPE)
    except FileNotFoundError:
        logger.error("File not found")
        sys.exit(1)
    return loaded_feature


def load_model(model_name: str, model_version: str) -> lightgbm.Booster:
    """モデルを読み込む関数"""
    try:
        load_model_path = "models:/" + model_name + "/" + model_version
        model = mlflow.lightgbm.load_model(load_model_path)
    except mlflow.exceptions.RestException:
        logger.error("Resource does not exist")
        sys.exit(1)
    return model


def save_prediction_result_csv(output_path: str, df_prediction: pd.DataFrame) -> None:
    """推論結果を保存する関数"""
    dt_now = datetime.now(timezone(timedelta(hours=+9), "JST"))
    dt_now_str = dt_now.strftime("%Y%m%d%H%M%S")
    out_name_pred = "misyu_score_list_" + dt_now_str + ".csv"
    try:
        output_file_path = Path(output_path) / out_name_pred
        df_prediction.to_csv(output_file_path, index=False, encoding="utf-8")
    except OSError:
        logger.error("OS Error")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("File not found")
        sys.exit(1)


def main() -> None:
    """main関数"""
    logger.info("---start predict---")
    # 入力パラメータの取得
    args = get_args()

    # inputデータの読み込み
    df_input = read_input_feature(args.feature_path)

    # モデルの読み込み
    model = load_model(args.model_name, args.model_version)

    # inputデータの整形
    list_index_col = ["LIV0EU_ガスメータ設置場所番号＿１ｘ", "LIV0EU_カスタマ番号＿８ｘ", "LIV0EU_使用契約番号＿４ｘ", "LIV0EU_支払契約番号＿２ｘ"]
    df_index = df_input[list_index_col].copy()
    df_index_drop = df_input.drop(columns=list_index_col)

    target_col = "未収フラグ"
    df_input = df_index_drop.drop(columns=[target_col])

    # 未収予測推論の実行
    predict_proba = model.predict(df_input, num_iteration=model.best_iteration)

    # 出力結果の整形
    df_prediction = pd.DataFrame({"index": df_input.index, "misyu_score": predict_proba})

    df_prediction = pd.concat([df_prediction, df_index], axis=1)

    # 推論結果の保存
    save_prediction_result_csv(args.output_path, df_prediction)


if __name__ == "__main__":
    main()
