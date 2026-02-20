# %%
!pip install emoji
!pip install -U bitsandbytes>=0.46.1

# %%
import pandas as pd
import emoji

# %%
def text_demojize(text):
    if pd.isna(text):
          return text
    return emoji.demojize(text, delimiters=(" :", ": "))

# %%
try:
    df1 = pd.read_csv('./data/processed/labelled_dataset.csv', engine='python')
    print('Displaying first 5 rows of labelled_dataset.csv:')
    display(df1.head())
except FileNotFoundError:
    print('Error: labelled_dataset.csv not found. Please check the file path.')
except Exception as e:
    print(f"An error occurred while reading labelled_dataset.csv: {e}")

# %%
try:
    df2 = pd.read_csv('./data/processed/labelled_dataset2.csv', engine='python') # Added engine='python'
    print('\nDisplaying first 5 rows of labelled_dataset2.csv:')
    display(df2.head())
except FileNotFoundError:
    print('Error: labelled_dataset2.csv not found. Please check the file path.')
except Exception as e:
    print(f"An error occurred while reading labelled_dataset2.csv: {e}")

# %%
df = pd.concat([df1, df2], ignore_index=True)

# %%
df_filtered = df[df['invalid'].isna()]

# %%
df_filtered.drop(columns=['invalid'], inplace=True)

# %%
df_filtered.columns

# %%
url_pattern = r'http\S+|www\.\S+'
mask = ~df_filtered['selftext'].str.contains(url_pattern, case=False, regex=True, na=False)
df_clean = df_filtered[mask]
df_clean.shape

# %%
df = df_clean

# %%
df['text'] = df['selftext'].apply(text_demojize)

# %%
df.columns

# %%
df = df.drop(['selftext', 'subreddit', "confidence"], axis=1)
df.to_csv('./data/final/final_dataset.csv', index=False)

# %%


# %%
target_columns = [
    'sadness', 'anxiety', 'rumination', 'self_focus',
    'hopelessness', 'emotional_volatility'
]
df['labels'] = df[target_columns].values.tolist()

# %%
# Keep only necessary columns and convert to Hugging Face Dataset
df = df[['text', 'labels']]
df = df.reset_index(drop=True)
df.to_csv('data_label.csv', index=False)

# %%



