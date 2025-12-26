from __future__ import annotations

import argparse
import os
import time
from typing import Literal

import pandas as pd


Provider = Literal["gemini", "openai"]


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def generate_with_gemini(text: str) -> str:
    api_key = _require_env("GOOGLE_API_KEY")
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("Install google-generativeai or choose another provider") from e

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Rewrite the following scientific abstract in different wording but keep the meaning. "
        "Do not add new facts. Output only the rewritten abstract.\n\n" + text
    )
    resp = model.generate_content(prompt)
    return str(resp.text or "").strip()


def generate_with_openai(text: str) -> str:
    api_key = _require_env("OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai or choose another provider") from e

    client = OpenAI(api_key=api_key)
    prompt = (
        "Rewrite the following scientific abstract in different wording but keep the meaning. "
        "Do not add new facts. Output only the rewritten abstract." 
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rewrite academic text."},
            {"role": "user", "content": prompt + "\n\n" + text},
        ],
        temperature=0.7,
    )
    return (r.choices[0].message.content or "").strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--sleep-s", type=float, default=0.5)
    args = p.parse_args()

    df = pd.read_csv(args.in_path)
    if "summary" not in df.columns:
        raise ValueError("Input CSV must have 'summary' column")

    df = df.dropna(subset=["summary"]).reset_index(drop=True)
    df = df.head(args.limit)

    out_rows = []
    for i, row in df.iterrows():
        src = str(row["summary"])
        if args.provider == "gemini":
            gen = generate_with_gemini(src)
        else:
            gen = generate_with_openai(src)

        out_rows.append({"id": row.get("id"), "title": row.get("title"), "summary": gen})
        time.sleep(args.sleep_s)

        if (i + 1) % 25 == 0:
            pd.DataFrame(out_rows).to_csv(args.out_path, index=False)

    pd.DataFrame(out_rows).to_csv(args.out_path, index=False)


if __name__ == "__main__":
    main()
