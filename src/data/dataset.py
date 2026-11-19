import csv
import gc
import io
import logging
from collections import Counter
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import requests
import urllib3
from datasets import (
    Dataset,
    Features,
    IterableDataset,
    Value,
    concatenate_datasets,
    load_dataset,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = Path(__file__).resolve().parent
REFERENCE_TABLE = DATA_DIR / "reference_table_bilingual.csv"
logger = logging.getLogger(__name__)


def get_raw_url(url: str) -> str:
    """Converts a GitHub blob URL to a raw content URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


LANGUAGES = [
    "amh", "ber", "chr", "grn", "haw", "ibo", "npi", "oci", "que", "yor", "zgh", "mus", "aed"
]

# High/mid-resource comparison languages. English has no bilingual data
HIGH_RESOURCE_LANGUAGES = ["eng", "deu", "spa", "hin", "zh"]


def get_language_name(lang_code: str) -> str:
    """
    Get full language name from language code using reference_table_bilingual.csv.
    e.g. 'npi' -> 'Nepali', 'chr' -> 'Cherokee'
    """
    lang_code = lang_code.strip().lower()
    with open(REFERENCE_TABLE, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["Code"].strip().lower() == lang_code:
                return row["Language"].strip()
    raise ValueError(f"Language code not found in {REFERENCE_TABLE}: {lang_code!r}")


mono_features = Features({"text": Value("string"), "origin": Value("string")})


def standardize_columns_mono(ds: Dataset | IterableDataset, language: str, lang_code: str, path: str):
    current_cols = ds.column_names
    if current_cols is None:
        ex = next(iter(ds))
        current_cols = list(ex.keys())
    assert isinstance(current_cols, list)

    col_lookup = {c.lower(): c for c in current_cols}

    if "source" in col_lookup:
        src_col = col_lookup["source"]
        if src_col != "origin":
            ds = ds.rename_column(src_col, "origin")
    else:
        ds = ds.map(lambda r: {"origin": path})

    if "text" not in col_lookup:
        search_cols = [
            language,
            language.lower(),
            lang_code,
            lang_code.lower(),
            lang_code.upper(),
            "Mayan",
            "Mayan language",
            "sentence",
            "text_sentence",
            "content",
            "Article",
            "Source",
            "Target",
            "inputs",
            "aed",
            "TUN",
            "Tunis",
            "tunisian",
            "Tunisian",
            "tn",
        ]
        for col in search_cols:
            matched_col = col_lookup.get(col.lower())
            if matched_col is not None and matched_col != "text":
                ds = ds.rename_column(matched_col, "text")
                break
        else:
            raise KeyError(
                f"Could not find matching text column for language={language!r} ({lang_code!r}) in columns={current_cols} (path={path!r})"
            )

    ds = ds.select_columns(["text", "origin"])
    return ds.cast(mono_features)


def standardize_columns_bi(ds: Dataset | IterableDataset, language: str, lang_code: str, path: str):
    if path == "KKKarim711/tunisian-english-parallel-pairs":
        logger.info("Columns: %s", list(next(iter(ds)).keys()))
        ds = ds.map(lambda r: {"origin": "KKKarim711/tunisian-english-parallel-pairs"})
        cols = set(next(iter(ds)).keys())
        for cand in ("english", "en", "src", "source"):
            if cand in cols and cand != "English":
                ds = ds.rename_column(cand, "English")
                break
        for cand in ("tn", "tunisian", "tunisian_arabic", "tunisian arabic", language.lower()):
            if cand in cols and cand != language:
                ds = ds.rename_column(cand, language)
                break
        ds = ds.select_columns(["English", language, "origin"])
        bi_features = Features({"English": Value("string"), language: Value("string"), "origin": Value("string")})
        return ds.cast(bi_features)

    current_cols = ds.column_names
    if current_cols is None:
        ex = next(iter(ds))
        current_cols = list(ex.keys())
    assert isinstance(current_cols, list)

    col_lookup = {c.lower(): c for c in current_cols}

    if "source" in col_lookup:
        src_col = col_lookup["source"]
        if src_col != "origin":
            ds = ds.rename_column(src_col, "origin")
    else:
        ds = ds.map(lambda r: {"origin": path})

    en_candidates = ["english", "en", "src", "source", "input", "inputs"]
    for cand in en_candidates:
        if cand in col_lookup:
            orig_en = col_lookup[cand]
            if orig_en != "English":
                ds = ds.rename_column(orig_en, "English")
            break
    else:
        raise KeyError(
            f"Could not find English column for language={language!r} in columns={current_cols} (path={path!r})"
        )

    lang_candidates = [
        language.lower(),
        lang_code.lower(),
        "tunisian",
        "tunisian_arabic",
        "tunisian arabic",
        "tun",
        "aeb",
        "aed",
        "tn",
        "creek",
        "muskogee",
        "muskogee (creek)",
        "Muskogee (Creek)",
        "mus",
        "tgt",
        "target",
        "output",
        "translation",
        "non_english",
    ]
    for cand in lang_candidates:
        if cand in col_lookup and col_lookup[cand] != "English":
            orig_lang = col_lookup[cand]
            if orig_lang != language:
                ds = ds.rename_column(orig_lang, language)
            break
    else:
        raise KeyError(
            f"Could not find target column for language={language!r} ({lang_code!r}) in columns={current_cols} (path={path!r})"
        )

    ds = ds.select_columns(["English", language, "origin"])
    bi_features = Features({"English": Value("string"), language: Value("string"), "origin": Value("string")})
    return ds.cast(bi_features)


def assemble_dataset(
    lang_code: str,
    type: Literal["mono", "bi"],
    tokenizer,
    max_samples: int | None = None,
    include_aya=True,
):
    file = "reference_table_monolingual.csv" if type == "mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    language = get_language_name(lang_code)

    dropped = 0

    def filter_one_tokens(row):
        nonlocal dropped
        if type == "mono":
            text = row["text"]
        else:
            text = row[language]
        if text is None:
            return False
        cond = len(tokenizer.tokenize(text)) > 1
        if not cond:
            dropped += 1
        return cond

    df = pd.read_csv(file_path)
    paths = df[(df["Language"] == language) & (df["hugging face"].notna())]
    dataset_list = []
    for _, row in paths.iterrows():
        path = row["hugging face"]
        if "aya_dataset" in str(path):
            continue
        assert isinstance(path, str)

        if path == "KKKarim711/tunisian-english-parallel-pairs":
            ds = load_dataset("KKKarim711/tunisian-english-parallel-pairs", split="train", streaming=True)
        elif str(path).startswith("http"):
            raw_url = get_raw_url(str(path))
            sep = "\t" if raw_url.endswith(".tsv") or "tatoeba" in raw_url.lower() else ","
            resp = requests.get(raw_url, verify=False)
            resp.raise_for_status()
            temp_df = pd.read_csv(io.BytesIO(resp.content), sep=sep)
            ds = Dataset.from_pandas(temp_df)
        else:
            repo = path
            config = None
            split_to_load = "train"
            if ":" in path:
                parts = path.split(":")
                repo = parts[0]
                config = parts[1]
                if len(parts) > 2:
                    split_to_load = parts[2]

            for split in [split_to_load, "full", lang_code]:
                try:
                    ds = load_dataset(repo, config, split=split, streaming=True)
                    assert isinstance(ds, IterableDataset)
                    break
                except Exception as e:
                    logger.warning(e)
                    continue
            else:
                raise ValueError(f"No split matching {[split_to_load, 'full', lang_code]} in {repo}")
            if repo == "Helsinki-NLP/opus-100":
                ds = ds.map(
                    lambda r: {
                        "English": r["translation"]["en"],
                        language: next(v for k, v in r["translation"].items() if k != "en"),
                    }
                )

        if type == "mono":
            ds = standardize_columns_mono(ds, language, lang_code, path)
        else:
            ds = standardize_columns_bi(ds, language, lang_code, path)

        dropped = 0
        ds = ds.filter(filter_one_tokens)
        if isinstance(ds, IterableDataset):
            if max_samples is None:
                stream = ds
            else:
                stream = ds.take(max_samples)
            ds = Dataset.from_list(list(stream))  # type:ignore
            del stream
            gc.collect()
        else:
            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))
        logger.info(f"Dropped {dropped} due to length")

        dataset_list.append(ds)

    if type == "mono" and include_aya:
        aya_dataset = load_dataset("CohereLabs/aya_dataset", split="train")
        aya_dataset = cast(Dataset, aya_dataset.filter(lambda x: x["language"].lower() == language.lower()))
        if len(aya_dataset) != 0:
            current_cols = aya_dataset.column_names
            search_cols = ["inputs", "targets", "text", "sentence"]
            for col in search_cols:
                if col in current_cols:
                    aya_dataset = aya_dataset.rename_column(col, "text")
                    break
            aya_dataset = aya_dataset.map(lambda r: {"origin": "CohereLabs/aya_dataset"})
            aya_dataset = aya_dataset.select_columns(["text", "origin"]).cast(mono_features)
            dataset_list.append(aya_dataset)

    if not dataset_list:
        raise ValueError(
            f"No datasets found for {language!r} (code={lang_code!r}, type={type!r}, matched_ref_rows={len(paths)}, include_aya={include_aya})"
        )
    dataset: Dataset = concatenate_datasets(dataset_list)
    if max_samples and max_samples > 0 and len(dataset) > max_samples:
        old_size = len(dataset)
        dataset = dataset.shuffle(42).select(range(max_samples))
        logger.info(f"Filtered full dataset from {old_size} to {max_samples} examples")
    logger.info(f"Data source breakdown: {Counter(dataset['origin'])}")
    splits = dataset.train_test_split(test_size=0.2, seed=42)

    if type == "bi":
        splits["test"] = splits["test"].select(range(min(1000, len(splits["test"]))))

    logger.info(f"Data splits: {splits}")
    return splits
