#!/usr/bin/env python3
"""Search server for synthetic multi-hop QA benchmark.

Contains a small, deterministic knowledge base of ~30 entities (people,
landmarks, countries). Supports keyword search via TF-IDF over the KB entries.

Every answer to make_multihop_data.py questions is guaranteed to be findable
via 1-2 search queries against this server. This makes it the ideal minimal
benchmark for comparing FoldAgent vs ContextGraph:
  - No external dependencies (no HuggingFace download, no Wikipedia)
  - Deterministic (same KB every run)
  - Fast startup (<1 second)
  - Agent can actually find useful info → non-zero reward signal

Runs on port 18999.
"""

import re
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from collections import Counter
import math

app = FastAPI()

# ── Knowledge Base ──

ENTITIES = {
    "Albert Einstein": {"born_in": "Ulm, Germany", "field": "physics", "nationality": "German-American", "known_for": "theory of relativity", "born_year": "1879"},
    "Marie Curie": {"born_in": "Warsaw, Poland", "field": "chemistry", "nationality": "Polish-French", "known_for": "radioactivity research", "born_year": "1867"},
    "Isaac Newton": {"born_in": "Woolsthorpe, England", "field": "physics", "nationality": "English", "known_for": "laws of motion", "born_year": "1643"},
    "Leonardo da Vinci": {"born_in": "Vinci, Italy", "field": "art", "nationality": "Italian", "known_for": "Mona Lisa", "born_year": "1452"},
    "Nikola Tesla": {"born_in": "Smiljan, Croatia", "field": "engineering", "nationality": "Serbian-American", "known_for": "alternating current", "born_year": "1856"},
    "Charles Darwin": {"born_in": "Shrewsbury, England", "field": "biology", "nationality": "English", "known_for": "theory of evolution", "born_year": "1809"},
    "Galileo Galilei": {"born_in": "Pisa, Italy", "field": "astronomy", "nationality": "Italian", "known_for": "telescope observations", "born_year": "1564"},
    "Ada Lovelace": {"born_in": "London, England", "field": "mathematics", "nationality": "English", "known_for": "first computer program", "born_year": "1815"},
    "Alexander Fleming": {"born_in": "Darvel, Scotland", "field": "medicine", "nationality": "Scottish", "known_for": "penicillin", "born_year": "1881"},
    "Frida Kahlo": {"born_in": "Coyoacán, Mexico", "field": "art", "nationality": "Mexican", "known_for": "self-portraits", "born_year": "1907"},
    "Alan Turing": {"born_in": "London, England", "field": "computer science", "nationality": "English", "known_for": "Turing machine", "born_year": "1912"},
    "Pythagoras": {"born_in": "Samos, Greece", "field": "mathematics", "nationality": "Greek", "known_for": "Pythagorean theorem", "born_year": "570 BC"},
}

LANDMARKS = {
    "Eiffel Tower": {"city": "Paris", "country": "France", "built_year": "1889", "type": "tower"},
    "Big Ben": {"city": "London", "country": "United Kingdom", "built_year": "1859", "type": "clock tower"},
    "Colosseum": {"city": "Rome", "country": "Italy", "built_year": "80 AD", "type": "amphitheater"},
    "Taj Mahal": {"city": "Agra", "country": "India", "built_year": "1653", "type": "mausoleum"},
    "Great Wall": {"city": "Beijing", "country": "China", "built_year": "7th century BC", "type": "wall"},
    "Statue of Liberty": {"city": "New York", "country": "United States", "built_year": "1886", "type": "statue"},
    "Sagrada Familia": {"city": "Barcelona", "country": "Spain", "built_year": "1882", "type": "basilica"},
    "Parthenon": {"city": "Athens", "country": "Greece", "built_year": "438 BC", "type": "temple"},
    "Sydney Opera House": {"city": "Sydney", "country": "Australia", "built_year": "1973", "type": "performing arts center"},
    "Christ the Redeemer": {"city": "Rio de Janeiro", "country": "Brazil", "built_year": "1931", "type": "statue"},
}

COUNTRIES = {
    "France": {"capital": "Paris", "continent": "Europe", "language": "French", "currency": "Euro", "population": "67 million"},
    "United Kingdom": {"capital": "London", "continent": "Europe", "language": "English", "currency": "Pound Sterling", "population": "67 million"},
    "Italy": {"capital": "Rome", "continent": "Europe", "language": "Italian", "currency": "Euro", "population": "60 million"},
    "India": {"capital": "New Delhi", "continent": "Asia", "language": "Hindi", "currency": "Indian Rupee", "population": "1.4 billion"},
    "China": {"capital": "Beijing", "continent": "Asia", "language": "Mandarin", "currency": "Yuan", "population": "1.4 billion"},
    "United States": {"capital": "Washington D.C.", "continent": "North America", "language": "English", "currency": "US Dollar", "population": "330 million"},
    "Spain": {"capital": "Madrid", "continent": "Europe", "language": "Spanish", "currency": "Euro", "population": "47 million"},
    "Greece": {"capital": "Athens", "continent": "Europe", "language": "Greek", "currency": "Euro", "population": "10 million"},
    "Australia": {"capital": "Canberra", "continent": "Oceania", "language": "English", "currency": "Australian Dollar", "population": "26 million"},
    "Brazil": {"capital": "Brasília", "continent": "South America", "language": "Portuguese", "currency": "Real", "population": "214 million"},
    "Germany": {"capital": "Berlin", "continent": "Europe", "language": "German", "currency": "Euro", "population": "83 million"},
    "Poland": {"capital": "Warsaw", "continent": "Europe", "language": "Polish", "currency": "Zloty", "population": "38 million"},
    "Croatia": {"capital": "Zagreb", "continent": "Europe", "language": "Croatian", "currency": "Euro", "population": "4 million"},
    "Mexico": {"capital": "Mexico City", "continent": "North America", "language": "Spanish", "currency": "Peso", "population": "130 million"},
    "Japan": {"capital": "Tokyo", "continent": "Asia", "language": "Japanese", "currency": "Yen", "population": "125 million"},
    "Scotland": {"capital": "Edinburgh", "continent": "Europe", "language": "English", "currency": "Pound Sterling", "population": "5 million"},
    "England": {"capital": "London", "continent": "Europe", "language": "English", "currency": "Pound Sterling", "population": "56 million"},
}

# Build document list for search
DOCS = []
DOCS_BY_ID = {}


def _build_docs():
    global DOCS, DOCS_BY_ID
    for name, info in ENTITIES.items():
        docid = f"person_{name.lower().replace(' ', '_')}"
        text = (f"{name} was born in {info['born_in']} in {info['born_year']}. "
                f"{name} was a {info['nationality']} {info['field']} expert, "
                f"best known for {info['known_for']}.")
        doc = {"docid": docid, "url": f"https://kb.example.com/{docid}", "text": text, "title": name}
        DOCS.append(doc)
        DOCS_BY_ID[docid] = doc

    for name, info in LANDMARKS.items():
        docid = f"landmark_{name.lower().replace(' ', '_')}"
        text = (f"The {name} is a famous {info['type']} located in {info['city']}, {info['country']}. "
                f"It was built in {info['built_year']}.")
        doc = {"docid": docid, "url": f"https://kb.example.com/{docid}", "text": text, "title": name}
        DOCS.append(doc)
        DOCS_BY_ID[docid] = doc

    for name, info in COUNTRIES.items():
        docid = f"country_{name.lower().replace(' ', '_')}"
        text = (f"{name} is a country in {info['continent']}. "
                f"Its capital is {info['capital']}. "
                f"The official language is {info['language']} and the currency is {info['currency']}. "
                f"Population: {info['population']}.")
        doc = {"docid": docid, "url": f"https://kb.example.com/{docid}", "text": text, "title": name}
        DOCS.append(doc)
        DOCS_BY_ID[docid] = doc


# TF-IDF index
IDF = {}
DOC_TFIDF = []


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def _build_index():
    global IDF, DOC_TFIDF
    N = len(DOCS)
    df = Counter()
    doc_tokens = []
    for doc in DOCS:
        tokens = tokenize(doc["text"] + " " + doc["title"])
        unique = set(tokens)
        for t in unique:
            df[t] += 1
        doc_tokens.append(tokens)

    IDF = {w: math.log(N / (c + 1)) for w, c in df.items()}

    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens) or 1
        DOC_TFIDF.append({w: (c / total) * IDF.get(w, 0) for w, c in tf.items()})


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None


@app.post("/search")
async def search(req: QueryRequest):
    query_tokens = tokenize(req.query)
    query_tf = Counter(query_tokens)
    total = len(query_tokens) or 1
    q_tfidf = {w: (c / total) * IDF.get(w, 0) for w, c in query_tf.items() if w in IDF}

    if not q_tfidf:
        return {"results": [{"docid": d["docid"], "url": d["url"], "text": d["text"]} for d in DOCS[:req.k]]}

    scores = []
    for i, dtf in enumerate(DOC_TFIDF):
        s = sum(q_tfidf.get(w, 0) * dtf.get(w, 0) for w in q_tfidf)
        if s > 0:
            scores.append((s, i))
    scores.sort(key=lambda x: -x[0])

    results = [{"docid": DOCS[i]["docid"], "url": DOCS[i]["url"], "text": DOCS[i]["text"]}
               for _, i in scores[:req.k]]
    if not results:
        results = [{"docid": d["docid"], "url": d["url"], "text": d["text"]} for d in DOCS[:req.k]]
    return {"results": results}


@app.post("/open")
async def open_page(req: OpenRequest):
    if req.docid and req.docid in DOCS_BY_ID:
        d = DOCS_BY_ID[req.docid]
        return {"results": [{"docid": d["docid"], "url": d["url"], "text": d["text"]}]}
    if req.url:
        for d in DOCS:
            if d["url"] == req.url:
                return {"results": [{"docid": d["docid"], "url": d["url"], "text": d["text"]}]}
    return {"results": [{"docid": DOCS[0]["docid"], "url": DOCS[0]["url"], "text": DOCS[0]["text"]}]}


if __name__ == "__main__":
    _build_docs()
    _build_index()
    print(f"MultiHop KB: {len(DOCS)} documents indexed. Starting server on port 18999...")
    uvicorn.run(app, host="0.0.0.0", port=18999, log_level="warning")
