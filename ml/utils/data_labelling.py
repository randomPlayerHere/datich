from google import genai
import os
from dotenv import load_dotenv
import pandas as pd
import json
import csv
import time
from itertools import cycle
import re
import concurrent.futures
from datetime import datetime

FAILED_JSON_PATH = "ml/data/processed/failed_json_outputs.jsonl"
CACHE_PATH = "ml/data/processed/labelled_dataset2.csv"
CHUNK_SIZE = 5


load_dotenv()
GEMINI_KEYS = os.getenv("GEMINI_KEYS").split(",")
key_cycle = cycle(GEMINI_KEYS)


def requestGemini(prompt, timeout=120, max_retries=5):
    last_error = None
    for _ in range(max_retries):
        api_key = next(key_cycle)
        try:
            client = genai.Client(api_key=api_key)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                response = future.result(timeout=timeout)
            return response.text
        except concurrent.futures.TimeoutError:
            print("⏱️ Gemini request timed out, rotating key...")
        except Exception as e:
            last_error = e
            print(f"⚠️ Gemini key failed ({e}), rotating key...")
        time.sleep(6)
    raise RuntimeError(f"All Gemini keys failed: {last_error}")


def promptGeneration(texts):
    items = []
    for i, text in enumerate(texts):
        items.append(f"{i+1}. (id={i}) {text}")
    joined = "\n".join(items)
    prompt = f"""
    You are an expert psychological text annotator.

  Your task is to quantitatively label mental-state dimensions from first-person text.
  This is NOT diagnosis. Do NOT infer mental disorders or give advice.
  Base scores ONLY on explicit evidence in the text.

  Each dimension must be scored from 0.0 to 1.0, where:
  - 0.1-0.2: mild, passing mention
  - 0.3-0.4: present but not dominant
  - 0.5-0.6: clearly present and important
  - 0.7-0.8: strong and dominant
  - 0.9-1.0: overwhelming, central theme


  Use low scores (<0.3) when evidence is weak or indirect.
  Be calibrated and consistent across all samples.

  Dimensions and definitions:

  - sadness:
    Low mood, grief, loss, emotional pain.
    Do not infer sadness without clear affective language.

  - anxiety:
    Worry, fear, tension, anticipation of harm or uncertainty.
    Do not confuse sadness with anxiety.

  - rumination:
    Repetitive, looping, or intrusive thinking.
    Requires explicit repetition or inability to stop thinking.

  - self_focus:
    Attention centered on the self, self-evaluation, self-worth.
    Does NOT require repetition.

  - hopelessness:
    Helplessness or negative expectations about the future.
    Do NOT infer hopelessness without future-oriented despair.

  - emotional_volatility:
    Emotional instability, rapid shifts, or intense swings.
    Keep low unless clear emotional fluctuation is present.

  Emotional volatility should be low unless clear emotional shifts or instability appear.
  Do not overuse high values.

  confidence:
  How confident you are that the assigned scores are accurate,
  based only on clarity and explicitness of the text.
  Low confidence if text is vague, short, or ambiguous.


  Do NOT provide explanations, interpretations, empathy, or advice.
  If you output anything other than pure JSON, the output is invalid.
  Do not include markdown fences.
  All numbers must be floats between 0.0 and 1.0.

  Return ONLY valid JSON.

  Format:
  [
    {{
      "id": 1,
      "scores": {{
        "sadness": 0.0,
        "anxiety": 0.0,
        "rumination": 0.0,
        "self_focus": 0.0,
        "hopelessness": 0.0,
        "emotional_volatility": 0.0
      }},
      "confidence": 0.0
    }}
  ]

  Treat each text independently.
  Do not compare emotional intensity between texts.
  Use the same internal calibration for all texts.
      Label the following texts independently:
    {joined}
    """
    return prompt


def safe_json_load(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON array found", text, 0)
    return json.loads(match.group())



def log_failed_json(chunk, raw_response, error):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "row_ids": chunk.index.tolist(),
        "selftext": chunk["selftext"].tolist(),
        "raw_response": raw_response,
        "error": str(error)
    }
    with open(FAILED_JSON_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def processedRowCount(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                next(reader)  # skip header
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    return 0


def labelChunk(chunk):
    prompt = promptGeneration(chunk["selftext"])
    response = requestGemini(prompt)
    labels = safe_json_load(response)  # may raise JSONDecodeError
    labels_df = pd.json_normalize(labels).sort_values("id").reset_index(drop=True)
    scores_df = labels_df.filter(regex="^scores\\.")
    scores_df.columns = scores_df.columns.str.replace("scores.", "", regex=False)
    labelled_chunk = pd.concat([chunk.reset_index(drop=True), scores_df, labels_df["confidence"]],axis=1)
    return labelled_chunk, response



def append_to_cache(df, cache_path):
    df.to_csv(
        cache_path,
        mode="a",
        header=not os.path.exists(cache_path),
        index=False
    )


def main(path="ml/data/processed/to_be_labelled2.csv"):
    processed_rows = processedRowCount(CACHE_PATH)
    print(f"Resuming from row {processed_rows}" if processed_rows else "Starting fresh")
    reader = pd.read_csv(path,chunksize=CHUNK_SIZE,skiprows=range(1, processed_rows + 1))
    for chunk_idx, chunk in enumerate(reader):
        try:
            labelled_chunk, response = labelChunk(chunk)
            append_to_cache(labelled_chunk, CACHE_PATH)
            print(f"Saved {len(labelled_chunk)} rows to cache")
            time.sleep(6)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in chunk {chunk_idx}, saving to failed file")
            log_failed_json(chunk=chunk,raw_response=response,error=e)

if __name__ == "__main__":
    main()