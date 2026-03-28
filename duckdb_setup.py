import duckdb

con = duckdb.connect("data/pipeline.duckdb")

# Preprocessing and crisis screening stages
con.execute("""
    CREATE TABLE IF NOT EXISTS preprocessed_posts (
        id          VARCHAR PRIMARY KEY,
        text        TEXT,
        subreddit   VARCHAR,
        author      VARCHAR,
        word_count  INTEGER,
        long_flag   BOOLEAN,
        crisis_flag INTEGER DEFAULT NULL  -- NULL = not screened, 0 = clean, 1 = crisis
    )
""")

# Labelled data and augmentation stages
con.execute("""
    CREATE TABLE IF NOT EXISTS labeled_posts (
        id                  VARCHAR PRIMARY KEY,
        text                TEXT,
        emotional_tone      FLOAT,
        energy_level        FLOAT,
        anxiety_signal      FLOAT,
        mood_stability      FLOAT,
        crisis_flag         INTEGER,
        annotator_agreement INTEGER,
        confidence          FLOAT,
        surface_mismatch    BOOLEAN,
        ambiguous           BOOLEAN,
        a1_tone      FLOAT, a1_energy   FLOAT, a1_anxiety  FLOAT, a1_stability FLOAT,
        a2_tone      FLOAT, a2_energy   FLOAT, a2_anxiety  FLOAT, a2_stability FLOAT,
        a3_tone      FLOAT, a3_energy   FLOAT, a3_anxiety  FLOAT, a3_stability FLOAT,
        reasoning           TEXT
    )
""")

con.close()
print("Database setup done")