import nltk
import numpy as np
import pandas as pd
from pathlib import Path

import os
from decouple import config as environ
from dotenv import load_dotenv

load_dotenv()

os.environ["ORCID_CODE"] = environ("ORCID_CODE")

nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")
wn = nltk.WordNetLemmatizer()
from dataclasses import asdict  # noqa

from bertopic import BERTopic  # noqa
from sentence_transformers import SentenceTransformer  # noqa
from umap import UMAP  # noqa

from my_scientific_profile.database.papers import (  # noqa
    load_all_papers_from_s3,
    convert_papers_to_dataframe,
    save_all_papers_to_s3,
)
from my_scientific_profile.papers.papers import Embedding  # noqa
from my_scientific_profile.database.aws_s3 import S3_BUCKET, S3_CLIENT  # noqa
from to_quarto.utils import ROOT_DIR

papers = load_all_papers_from_s3(s3_client=S3_CLIENT, s3_bucket=S3_BUCKET)
df = convert_papers_to_dataframe(papers)
stopwords = nltk.corpus.stopwords.words("english")
df["abstract"] = df["abstract"].fillna("")

# 1. Fallback to title if abstract is empty or effectively empty (e.g., < 20 chars)
df["text_to_embed"] = df.apply(
    lambda x: x["title"] if (not x["abstract"] or len(str(x["abstract"]).strip()) < 20) 
    else x["abstract"], 
    axis=1
)

# 2. Clean the combined text
df["text_lemmatized"] = df["text_to_embed"].apply(
    lambda x: " ".join([
        wn.lemmatize(w.lower()) 
        for w in str(x).split() 
        if w.lower() not in stopwords and w.isalnum()
    ])
)

umap_model = UMAP(
    n_neighbors=5, n_components=2, min_dist=0.0, metric="cosine", random_state=100
)

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = sentence_model.encode(df["text_lemmatized"].tolist())

topic_model = BERTopic(
    umap_model=umap_model,
    min_topic_size=2,
).fit(df["text_lemmatized"], embeddings=embeddings)

topic_labels = topic_model.generate_topic_labels(
    nr_words=3, topic_prefix=False, word_length=15, separator=" | "
)
topic_model.set_topic_labels(topic_labels)
df["topic"] = topic_model.topics_
plotly_obj = topic_model.visualize_documents(
    docs=df.index,
    embeddings=embeddings,
    hide_annotations=False,
    custom_labels=True,
    title="Literature graph",
)
df_coord = pd.json_normalize(
    [
        {"paper_id": int(index), "x": x, "y": y}
        for d in plotly_obj.data
        for index, x, y in zip(d["hovertext"], d["x"], d["y"])
        if not np.isnan(index)
    ]
)
df_coord["topic"] = df.iloc[df_coord["paper_id"].values].topic.values
topic_keys = list(topic_model.get_topics().keys())
df_coord["topic_name"] = df_coord.apply(lambda x: topic_labels[topic_keys.index(x["topic"])], axis=1)
df_coord["title"] = df_coord.apply(
    lambda x: f"{df.iloc[x['paper_id']].title[:50] + '...'}"
    if len(df.iloc[x["paper_id"]].title) > 50
    else df.iloc[x["paper_id"]].title,
    axis=1,
)
df_coord["doi"] = df.iloc[df_coord["paper_id"]].doi.values
for _, item in df_coord.iterrows():
    paper = [p for p in papers if p.doi == item["doi"]][0]
    paper.embedding = Embedding(
        x=item["x"],
        y=item["y"],
        topic_number=item["topic"],
        topic_name=item["topic_name"],
    )

save_all_papers_to_s3(S3_CLIENT, S3_BUCKET)


df = convert_papers_to_dataframe(papers)

path = Path(ROOT_DIR)
team_path = path.joinpath("data")
df.to_json(team_path.joinpath("all_papers.json"))
df.to_csv(team_path.joinpath("all_papers.csv"))
