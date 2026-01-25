from google import genai
import os
from dotenv import load_dotenv
import pandas as pd
import json
import time

load_dotenv()

def requestGemini(prompt):
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    return response.text


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
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    return json.loads(text)


def processedRowCount(cache_path):
    if os.path.exists(cache_path):
        return len(pd.read_csv(cache_path))
    return 0


def labelChunk(chunk):
    prompt= promptGeneration(chunk["selftext"])
    response= requestGemini(prompt)
    labels = safe_json_load(response) #stores the list of dicts
    labels_df =pd.json_normalize(labels).sort_values("id").reset_index(drop = True) #convert list of dicts to dataframe
    scores_df = labels_df.filter(regex = "^scores\\.") #extract only the scores columns
    scores_df.columns= scores_df.columns.str.replace("scores.", "", regex=False) #rename columns to remove "scores." prefix
    labelled_chunk =pd.concat(
        [chunk.reset_index(drop=True),
            scores_df,
            labels_df["confidence"]],axis=1)
    return labelled_chunk


def append_to_cache(df, cache_path):
    df.to_csv(
        cache_path,
        mode="a",
        header=not os.path.exists(cache_path),
        index=False
    )



def main(path="ml/data/processed/to_be_labelled.csv"):
    CACHE_PATH = "ml/data/processed/labelled_cache.csv"
    CHUNK_SIZE = 5
    processed_rows = processedRowCount(CACHE_PATH)
    print(f"Resuming from row {processed_rows}" if processed_rows else "Starting fresh")
    reader = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        skiprows=range(1, processed_rows + 1)
    )
    for chunk in reader:
        try:
            labelled_chunk = labelChunk(chunk)
            append_to_cache(labelled_chunk, CACHE_PATH)
            print(f"Saved {len(labelled_chunk)} rows to cache")
            time.sleep(2)  # brief pause to respect rate limits
        except json.JSONDecodeError:
            print("JSON parsing error, skipping chunk")

if __name__ == "__main__":
    main()
    
