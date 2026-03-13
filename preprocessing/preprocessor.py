from file_merger import getData
import re
import emoji
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class Preprocessor:
    def __init__(self, path = "data/raw", sample_size = 5000):
        self.df = getData(path, sample_size)
        self.df.rename(columns={'selftext': 'text'}, inplace=True)


    def stripText(self, text):
        if not isinstance(text, str): #filtered by too short in filterData function
            return ""
        text = re.sub(r'u/\w+', '', text) # User mentions
        text = re.sub(r'r/\w+', '', text) #subreddit mentions
        text = re.sub(r'http\S+|www\.\S+', '', text) # urls
        text = re.sub(r'\[.*?\]', '', text) # markdown links
        boilerplate = r'\b(Edit|TL;DR|UPDATE|TLDR)\s*:.*'
        text = re.sub(boilerplate, '', text, flags=re.IGNORECASE) # boilerplate lines
        text = re.sub(r'\*{1,2}|_{1,2}', '', text) #bold and italic markdown
        text = re.sub(r'^\s*>+', '', text, flags=re.MULTILINE) #bloackquotes markdown
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE) #horizontal lines remove
        text = re.sub(r'\s+', ' ', text).strip() #extra whitespace remove
        text = emoji.demojize(text, delimiters=(" ", " "))
        return text
    

    def filterData(self):
        self.df['word_count'] = self.df['text'].apply(lambda x: len(x.split()))
        too_Short = self.df[self.df['word_count'] < 20]
        too_Long = self.df[self.df['word_count'] > 600]
        self.df = self.df[self.df["word_count"] >= 20]
        self.df["long_flag"] = self.df["word_count"] > 600

        self.df = self.df.drop_duplicates(subset=["text"])

        def closeDuplicatesDrop(self, threshold = 0.85):
            to_drop = set()
            for user, group in self.df.groupby('author'):
                if len(group) <2:
                    continue
                indices = group.index.tolist()
                texts = group['text'].tolist()
                tfidf = TfidfVectorizer().fit_transform(texts)
                """
                tfiff matrix example:
                         text1   text2   text3
                word1    0.5    0.5     0.0
                word2    0.0    0.0     1.0
                word3    0.7    0.7     0.0
                """
                sims = cosine_similarity(tfidf)
                """
                sims matrix example:
                         text1   text2   text3
                text1     1.00   0.97    0.40
                text2     0.97   1.00    0.38
                text3     0.40   0.38    1.00
                """
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if sims[i][j] >= threshold:
                            to_drop.add(indices[j])
            return self.df.drop(index=to_drop)
        if 'author' in self.df.columns:
            self.df = closeDuplicatesDrop(self) 
        
        def isEnglish(text):
            try:
                return detect(text) == "en"
            except:
                return False  
        self.df = self.df[self.df["text"].apply(isEnglish)] 
        

    def questionCheckFilter(self):
        QUESTION_PATTERNS = [
            r"^(does anyone|do you|can anyone|has anyone|where can|who knows|what is|how do|is there|are there)",
            r"^(looking for|anyone know|any recommendations|can someone|could someone)",
        ]

        EMOTIONAL_KEYWORDS = [
            "feel", "feeling", "felt", "emotion", "sad", "depressed", "anxious",
            "scared", "hopeless", "numb", "exhausted", "overwhelmed", "lonely",
            "worthless", "happy", "hopeful", "grateful", "relieved", "angry",
            "frustrated", "hurt", "lost", "broken", "empty", "afraid", "worried"
        ]
        def is_pure_question(text):
            text_lower = text.lower().strip()
            is_question = (
                text_lower.endswith("?") or
                any(re.match(p, text_lower) for p in QUESTION_PATTERNS)
            )
            if not is_question:
                return False
            has_emotion = any(kw in text_lower for kw in EMOTIONAL_KEYWORDS)
            return not has_emotion
        self.df = self.df[~self.df["text"].apply(is_pure_question)]

    def preprocess(self):
        self.df['text'] = self.df['text'].apply(self.stripText)
        self.filterData()
        self.questionCheckFilter()
        return self.df

