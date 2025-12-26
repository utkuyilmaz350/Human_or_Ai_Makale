from __future__ import annotations

import argparse

import pandas as pd

from human_or_ai.data_clean.clean import normalize_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--human", required=True, help="CSV: arxiv human abstracts")
    p.add_argument("--ai", required=True, help="CSV: ai generated abstracts")
    p.add_argument("--out", required=True, help="output dataset csv")
    args = p.parse_args()

    human = pd.read_csv(args.human)
    ai = pd.read_csv(args.ai)

    human = human[["summary"]].rename(columns={"summary": "text"})
    human["label"] = "human"

    ai = ai[["summary"]].rename(columns={"summary": "text"})
    ai["label"] = "ai"

    df = pd.concat([human, ai], axis=0, ignore_index=True)
    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df[df["text"].str.len() >= 20]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
