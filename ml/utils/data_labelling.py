from google import genai
import os
from dotenv import load_dotenv
import pandas as pd
import json

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
            
# def promptGeneration(chnk : pd.DataFrame):
#     prompt = f"""

# Label the following texts independently:

# 1. {chnk.iloc[0]}
# 2. {chnk.iloc[1]}
# 3. {chnk.iloc[2]}
# 4. {chnk.iloc[3]}
# 5. {chnk.iloc[4]}
# """

#     return prompt


def safe_json_load(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    return json.loads(text)



def main(path="ml/data/processed/to_be_labelled.csv"):
    all_labelled_chunks = []
    for chunk in pd.read_csv(path,chunksize=5):
        print(chunk['selftext'])
        prompt = promptGeneration(chunk['selftext'])
        response = requestGemini(prompt)
        try:
            labels = safe_json_load(response)
        except json.JSONDecodeError:
            print("JSON parsing error, skipping chunk")
            continue
        labels_df = pd.json_normalize(labels).sort_values("id").reset_index(drop=True)
        scores_df = labels_df.filter(regex="^scores\\.")
        scores_df.columns = scores_df.columns.str.replace("scores.", "", regex=False)
        labelled_chunk = pd.concat(
            [chunk.reset_index(drop=True), scores_df, labels_df["confidence"]],
            axis=1
        )
        print(labelled_chunk.columns)
        print(type(labelled_chunk))
        all_labelled_chunks.append(labelled_chunk)

if __name__ == "__main__":
    main()
    
