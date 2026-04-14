#!/usr/bin/env python3
"""Mock search server for FoldAgent / ContextGraph training tests.

Returns fake search results that contain the answers to simple QA questions.
This lets the agent loop run without a real corpus, while still producing
non-zero reward signals during early RL steps.

Runs on port 18999.
"""

import re
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Knowledge base: each entry has keywords + a doc with the answer.
# When a query matches keywords, the corresponding doc is returned.
KNOWLEDGE_BASE = [
    # Capitals
    {"keywords": ["france", "french", "paris"], "doc": ("doc_paris", "https://example.com/paris",
        "Paris is the capital and largest city of France, located on the River Seine.")},
    {"keywords": ["japan", "japanese", "tokyo"], "doc": ("doc_tokyo", "https://example.com/tokyo",
        "Tokyo is the capital city of Japan, one of the most populous metropolitan areas in the world.")},
    {"keywords": ["italy", "italian", "rome"], "doc": ("doc_rome", "https://example.com/rome",
        "Rome is the capital city of Italy and a major historical and cultural center.")},
    {"keywords": ["germany", "german", "berlin"], "doc": ("doc_berlin", "https://example.com/berlin",
        "Berlin is the capital of Germany and its largest city.")},
    {"keywords": ["spain", "spanish", "madrid"], "doc": ("doc_madrid", "https://example.com/madrid",
        "Madrid is the capital and most populous city of Spain.")},
    {"keywords": ["russia", "russian", "moscow"], "doc": ("doc_moscow", "https://example.com/moscow",
        "Moscow is the capital and largest city of Russia, located on the Moskva River.")},
    {"keywords": ["china", "chinese", "beijing"], "doc": ("doc_beijing", "https://example.com/beijing",
        "Beijing is the capital city of China, with over 21 million residents.")},
    {"keywords": ["australia", "australian", "canberra"], "doc": ("doc_canberra", "https://example.com/canberra",
        "Canberra is the capital city of Australia, located in the Australian Capital Territory.")},
    # Famous people
    {"keywords": ["romeo", "juliet", "shakespeare"], "doc": ("doc_shakespeare", "https://example.com/shakespeare",
        "William Shakespeare was an English playwright who wrote Romeo and Juliet between 1594 and 1596.")},
    {"keywords": ["mona", "lisa", "vinci", "leonardo"], "doc": ("doc_vinci", "https://example.com/vinci",
        "The Mona Lisa was painted by Leonardo da Vinci, the famous Italian Renaissance polymath.")},
    {"keywords": ["relativity", "einstein"], "doc": ("doc_einstein", "https://example.com/einstein",
        "Albert Einstein developed the theory of relativity, revolutionizing modern physics.")},
    {"keywords": ["first", "president", "united states", "washington"], "doc": ("doc_washington", "https://example.com/washington",
        "George Washington was the first president of the United States, serving from 1789 to 1797.")},
    {"keywords": ["1984", "novel", "orwell"], "doc": ("doc_orwell", "https://example.com/orwell",
        "George Orwell wrote the novel 1984, a dystopian classic published in 1949.")},
    {"keywords": ["telephone", "invent", "bell"], "doc": ("doc_bell", "https://example.com/bell",
        "Alexander Graham Bell invented the telephone in 1876.")},
    {"keywords": ["ninth", "symphony", "beethoven"], "doc": ("doc_beethoven", "https://example.com/beethoven",
        "Ludwig van Beethoven composed the Ninth Symphony, completed in 1824.")},
    {"keywords": ["penicillin", "fleming"], "doc": ("doc_fleming", "https://example.com/fleming",
        "Alexander Fleming discovered penicillin in 1928.")},
    # Science / nature
    {"keywords": ["largest", "planet", "jupiter", "solar"], "doc": ("doc_jupiter", "https://example.com/jupiter",
        "Jupiter is the largest planet in our solar system.")},
    {"keywords": ["smallest", "planet", "mercury"], "doc": ("doc_mercury", "https://example.com/mercury",
        "Mercury is the smallest planet in our solar system, closest to the Sun.")},
    {"keywords": ["gold", "chemical", "symbol", "au"], "doc": ("doc_gold", "https://example.com/gold",
        "The chemical symbol for gold is Au, derived from the Latin word aurum.")},
    {"keywords": ["water", "chemical", "h2o"], "doc": ("doc_water", "https://example.com/water",
        "The chemical formula for water is H2O, consisting of two hydrogen atoms and one oxygen.")},
    {"keywords": ["continent", "earth", "seven"], "doc": ("doc_continents", "https://example.com/continents",
        "There are 7 continents on Earth: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.")},
    {"keywords": ["tallest", "mountain", "everest"], "doc": ("doc_everest", "https://example.com/everest",
        "Mount Everest is the tallest mountain in the world, at 8,848.86 meters above sea level.")},
    {"keywords": ["longest", "river", "nile"], "doc": ("doc_nile", "https://example.com/nile",
        "The Nile is the longest river in the world, flowing through northeastern Africa.")},
    {"keywords": ["largest", "ocean", "pacific"], "doc": ("doc_pacific", "https://example.com/pacific",
        "The Pacific Ocean is the largest and deepest ocean on Earth.")},
    # History
    {"keywords": ["world war", "ii", "ended", "1945"], "doc": ("doc_ww2", "https://example.com/ww2",
        "World War II ended in 1945 with the surrender of Germany in May and Japan in September.")},
    {"keywords": ["berlin", "wall", "fall", "1989"], "doc": ("doc_berlinwall", "https://example.com/berlinwall",
        "The Berlin Wall fell in 1989, marking the end of the Cold War era.")},
    {"keywords": ["moon", "first", "armstrong"], "doc": ("doc_armstrong", "https://example.com/armstrong",
        "Neil Armstrong was the first person to walk on the Moon on July 20, 1969.")},
    {"keywords": ["titanic", "sink", "1912"], "doc": ("doc_titanic", "https://example.com/titanic",
        "The Titanic sank on April 15, 1912, after striking an iceberg in the North Atlantic.")},
    # Sports / pop culture
    {"keywords": ["soccer", "team", "field", "11", "players"], "doc": ("doc_soccer", "https://example.com/soccer",
        "A soccer team has 11 players on the field, including one goalkeeper.")},
    {"keywords": ["king", "sport", "soccer", "football"], "doc": ("doc_kingsport", "https://example.com/kingsport",
        "Soccer (also known as football) is often called the king of sports due to its global popularity.")},
    {"keywords": ["harry", "potter", "main"], "doc": ("doc_harry", "https://example.com/harry",
        "Harry Potter is the main character of the Harry Potter book series by J.K. Rowling.")},
    {"keywords": ["wizard", "school", "hogwarts"], "doc": ("doc_hogwarts", "https://example.com/hogwarts",
        "Hogwarts School of Witchcraft and Wizardry is the wizarding school in the Harry Potter series.")},
]

# Build flat doc index by docid for /open endpoint
DOCS_BY_ID = {entry["doc"][0]: entry["doc"] for entry in KNOWLEDGE_BASE}


class QueryRequest(BaseModel):
    query: str
    k: int = 10


class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None


def _to_doc(t):
    docid, url, text = t
    return {"docid": docid, "url": url, "text": text}


def _score(query_words, keywords):
    return sum(1 for kw in keywords if kw in query_words)


@app.post("/search")
async def search(req: QueryRequest):
    """Score each KB entry against the query, return top-k."""
    query_words = set(re.findall(r"\w+", req.query.lower()))
    scored = []
    for entry in KNOWLEDGE_BASE:
        s = _score(query_words, entry["keywords"])
        if s > 0:
            scored.append((s, entry["doc"]))
    # Sort by score desc
    scored.sort(key=lambda x: -x[0])
    results = [_to_doc(t) for _, t in scored[:req.k]]
    # Fallback: if nothing matched, return first 3 entries
    if not results:
        results = [_to_doc(entry["doc"]) for entry in KNOWLEDGE_BASE[:3]]
    return {"results": results}


@app.post("/open")
async def open_page(req: OpenRequest):
    if req.docid and req.docid in DOCS_BY_ID:
        return {"results": [_to_doc(DOCS_BY_ID[req.docid])]}
    if req.url:
        for entry in KNOWLEDGE_BASE:
            if entry["doc"][1] == req.url:
                return {"results": [_to_doc(entry["doc"])]}
    # Fallback
    return {"results": [_to_doc(KNOWLEDGE_BASE[0]["doc"])]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18999, log_level="warning")
