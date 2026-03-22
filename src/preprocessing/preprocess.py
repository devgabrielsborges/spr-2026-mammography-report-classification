import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ACHADOS_PATTERNS = {
    "achado_calcif_benignas": r"[Cc]alcificações benignas",
    "achado_sem_calcif_suspeitas": (
        r"[Nn]ão se observam (?:calcificações suspeitas"
        r"|microcalcificações pleomórficas|calcificações pleomórficas)"
    ),
    "achado_axilares_normais": r"regiões? axilares? não apresenta",
    "achado_mamas_lipossubst": r"[Mm]amas? parcialmente lipossubstituídas",
    "achado_mamas_densas_fibro": r"[Mm]amas? com densidades fibroglandulares",
    "achado_linfonodo_intramamario": r"[Ll]infonodo[s]? intramamário",
    "achado_assimetria": r"[Aa]ssimetria",
    "achado_nodulo": r"[Nn]ódulo",
    "achado_distorcao_arquitetural": r"[Dd]istorção arquitetural",
    "achado_calcif_vasculares": r"calcificações.*vasculares|vasculares",
    "achado_espessamento_cutaneo": r"[Ee]spessamento cutâneo",
    "achado_retracao": r"[Rr]etração",
}


def _extract_indicacao(text: str) -> str | None:
    m = re.search(
        r"Indicação clínica:\s*[\n\r]+\s*(.*?)"
        r"(?=[\n\r]+(?:Achados|Realizad|Mamas|Tecido|Calc|Impl|Alt|Desc))",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _classify_indicacao(text: str | None) -> str:
    if pd.isna(text) or text is None:
        return "desconhecido"
    t = text.lower().replace("\n", " ").replace("\r", " ").strip().rstrip(".")
    if "reavaliação" in t or "reavaliaçao" in t:
        return "reavaliacao"
    if "rastreamento" in t and "controle" in t:
        return "rastreamento_controle"
    if "rastreamento" in t:
        return "rastreamento"
    if "controle" in t:
        return "controle"
    if "primeiro exame" in t or "primeira mamografia" in t:
        return "primeiro_exame"
    if "rotina" in t:
        return "rotina"
    if "sintomática" in t or "nódulo palpável" in t or "queixa" in t:
        return "sintomatica"
    return "outro"


def _extract_achados(text: str) -> str | None:
    m = re.search(
        r"Achados:\s*[\n\r]+(.*?)(?=[\n\r]+Análise comparativa:|$)",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"Indicação clínica:.*?[\n\r]+.*?\.[\n\r]+(.*?)"
        r"(?=[\n\r]+Análise comparativa:|$)",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _extract_analise_comparativa(text: str) -> str | None:
    m = re.search(r"Análise comparativa:\s*[\n\r]+(.*?)$", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured features from the raw report text."""
    report = df["report"]

    indicacao_raw = report.apply(_extract_indicacao)
    df["indicacao_class"] = indicacao_raw.apply(_classify_indicacao)

    achados = report.apply(_extract_achados)
    for col_name, pattern in ACHADOS_PATTERNS.items():
        df[col_name] = achados.str.contains(pattern, na=False).astype(int)

    df["analise_comparativa"] = report.apply(_extract_analise_comparativa)

    return df


def preprocess_data(data: pd.DataFrame, target_column: str | None = None):
    target_column = target_column or os.getenv("TARGET_COLUMN", "target")
    id_column = os.getenv("ID_COLUMN", "ID")

    data = extract_features(data)

    achado_cols = list(ACHADOS_PATTERNS.keys())
    feature_cols = ["indicacao_class", "analise_comparativa"] + achado_cols
    X = data[feature_cols].copy()
    y = data[target_column].astype("uint8") if target_column in data.columns else None

    numerical_columns = achado_cols
    categorical_columns = ["indicacao_class"]
    text_columns = ["analise_comparativa"]

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    text_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
            ("txt", text_transformer, text_columns),
        ]
    )

    output_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        X_train_out = (
            X_train_preprocessed.toarray()
            if sparse.issparse(X_train_preprocessed)
            else X_train_preprocessed
        )
        X_test_out = (
            X_test_preprocessed.toarray()
            if sparse.issparse(X_test_preprocessed)
            else X_test_preprocessed
        )
        np.save(output_dir / "X_train_preprocessed.npy", X_train_out)
        np.save(output_dir / "X_test_preprocessed.npy", X_test_out)
        np.save(output_dir / "y_train.npy", y_train.values)
        np.save(output_dir / "y_test.npy", y_test.values)
    else:
        X_preprocessed = preprocessor.fit_transform(X)
        X_out = (
            X_preprocessed.toarray()
            if sparse.issparse(X_preprocessed)
            else X_preprocessed
        )
        np.save(output_dir / "X_preprocessed.npy", X_out)

    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    submission_test_path = raw_dir / "test.csv"
    if submission_test_path.exists():
        submission_data = pd.read_csv(submission_test_path)
        if "report" in submission_data.columns:
            submission_data = extract_features(submission_data)
            X_submission = submission_data[feature_cols].copy()
            X_submission_preprocessed = preprocessor.transform(X_submission)
            X_sub_out = (
                X_submission_preprocessed.toarray()
                if sparse.issparse(X_submission_preprocessed)
                else X_submission_preprocessed
            )
            np.save(output_dir / "X_submission_preprocessed.npy", X_sub_out)


if __name__ == "__main__":
    load_dotenv(override=True)
    data = pd.read_csv(Path(os.getenv("DATA_RAW_DIR", "data/raw")) / "train.csv")
    preprocess_data(data)
    print("Data preprocessed successfully")
