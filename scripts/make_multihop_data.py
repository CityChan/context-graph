#!/usr/bin/env python3
"""Generate synthetic multi-hop QA dataset for FoldAgent / ContextGraph benchmarking.

Each question requires 2-3 search steps to answer. The answers are guaranteed
to exist in the companion knowledge base (multihop_search_server.py).

Two question types:
  - **bridge**: sequential hops (A→B→answer). e.g. "What is the capital of the
    country where the Eiffel Tower is located?" → France → Paris
  - **comparison**: parallel hops (search A, search B, compare). e.g. "Are the
    Eiffel Tower and Big Ben in the same country?" → France vs UK → No

This is designed to be the minimal benchmark showing ContextGraph's value:
  - comparison questions benefit from cross-branch graph ops (add_edge, merge)
  - bridge questions test sequential reasoning
  - All answers are in the KB → ~40-70% accuracy expected → strong gradient signal
  - No external API needed, no large corpus, runs on 1 GPU
"""

import os
import json
import argparse
import pandas as pd
import random

# ── Knowledge Base (same as multihop_search_server.py) ──

ENTITIES = {
    # People
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
}


def generate_bridge_questions():
    """Generate 2-3 hop bridge questions (A→B→answer)."""
    questions = []

    # Type 1: landmark → country → property
    for lname, linfo in LANDMARKS.items():
        country = linfo["country"]
        if country in COUNTRIES:
            cinfo = COUNTRIES[country]
            questions.append({
                "question": f"What is the capital of the country where the {lname} is located?",
                "answer": cinfo["capital"],
                "type": "bridge",
                "hops": 2,
                "reasoning": f"{lname} → {country} → capital: {cinfo['capital']}",
            })
            questions.append({
                "question": f"What language is spoken in the country where the {lname} is located?",
                "answer": cinfo["language"],
                "type": "bridge",
                "hops": 2,
                "reasoning": f"{lname} → {country} → language: {cinfo['language']}",
            })
            questions.append({
                "question": f"What continent is the {lname} on?",
                "answer": cinfo["continent"],
                "type": "bridge",
                "hops": 2,
                "reasoning": f"{lname} → {country} → continent: {cinfo['continent']}",
            })

    # Type 2: person → birthplace → country → property
    for pname, pinfo in ENTITIES.items():
        born_parts = pinfo["born_in"].split(", ")
        if len(born_parts) >= 2:
            country_name = born_parts[-1]
            if country_name in COUNTRIES:
                cinfo = COUNTRIES[country_name]
                questions.append({
                    "question": f"What is the capital of the country where {pname} was born?",
                    "answer": cinfo["capital"],
                    "type": "bridge",
                    "hops": 2,
                    "reasoning": f"{pname} → born in {pinfo['born_in']} → {country_name} → capital: {cinfo['capital']}",
                })
                questions.append({
                    "question": f"What currency is used in the country where {pname} was born?",
                    "answer": cinfo["currency"],
                    "type": "bridge",
                    "hops": 2,
                    "reasoning": f"{pname} → born in {pinfo['born_in']} → {country_name} → currency: {cinfo['currency']}",
                })

    # Type 3: 3-hop: person → birth city → find landmark in same city
    for pname, pinfo in ENTITIES.items():
        born_city = pinfo["born_in"].split(",")[0].strip()
        for lname, linfo in LANDMARKS.items():
            if linfo["city"] == born_city:
                questions.append({
                    "question": f"What famous landmark is in the same city where {pname} was born?",
                    "answer": lname,
                    "type": "bridge",
                    "hops": 3,
                    "reasoning": f"{pname} → born in {born_city} → landmark in {born_city}: {lname}",
                })

    return questions


def generate_comparison_questions():
    """Generate comparison questions (search A, search B, compare)."""
    questions = []
    people = list(ENTITIES.items())
    landmarks = list(LANDMARKS.items())

    # Type 1: Are two people from the same country?
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1_name, p1_info = people[i]
            p2_name, p2_info = people[j]
            c1 = p1_info["born_in"].split(", ")[-1]
            c2 = p2_info["born_in"].split(", ")[-1]
            same = "Yes" if c1 == c2 else "No"
            questions.append({
                "question": f"Were {p1_name} and {p2_name} born in the same country?",
                "answer": same,
                "type": "comparison",
                "hops": 2,
                "reasoning": f"{p1_name} → {c1}, {p2_name} → {c2} → {same}",
            })

    # Type 2: Are two people in the same field?
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1_name, p1_info = people[i]
            p2_name, p2_info = people[j]
            same = "Yes" if p1_info["field"] == p2_info["field"] else "No"
            questions.append({
                "question": f"Are {p1_name} and {p2_name} known for work in the same field?",
                "answer": same,
                "type": "comparison",
                "hops": 2,
                "reasoning": f"{p1_name} → {p1_info['field']}, {p2_name} → {p2_info['field']} → {same}",
            })

    # Type 3: Are two landmarks in the same country?
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            l1_name, l1_info = landmarks[i]
            l2_name, l2_info = landmarks[j]
            same = "Yes" if l1_info["country"] == l2_info["country"] else "No"
            questions.append({
                "question": f"Are the {l1_name} and the {l2_name} in the same country?",
                "answer": same,
                "type": "comparison",
                "hops": 2,
                "reasoning": f"{l1_name} → {l1_info['country']}, {l2_name} → {l2_info['country']} → {same}",
            })

    # Type 4: Who was born earlier?
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1_name, p1_info = people[i]
            p2_name, p2_info = people[j]
            y1 = p1_info["born_year"].replace(" BC", "")
            y2 = p2_info["born_year"].replace(" BC", "")
            try:
                v1 = -int(y1) if "BC" in p1_info["born_year"] else int(y1)
                v2 = -int(y2) if "BC" in p2_info["born_year"] else int(y2)
                earlier = p1_name if v1 < v2 else p2_name
                questions.append({
                    "question": f"Who was born earlier, {p1_name} or {p2_name}?",
                    "answer": earlier,
                    "type": "comparison",
                    "hops": 2,
                    "reasoning": f"{p1_name} → {p1_info['born_year']}, {p2_name} → {p2_info['born_year']} → {earlier}",
                })
            except ValueError:
                pass

    return questions


def to_row(item, workflow):
    return {
        "prompt": [{"role": "user", "content": item["question"]}],
        "ability": "LocalSearch@multihop",
        "extra_info": {
            "query": item["question"],
            "answer": item["answer"],
            "problem_statement": item["question"],
            "type": item["type"],
            "hops": item["hops"],
            "reasoning": item["reasoning"],
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_val", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    bridge = generate_bridge_questions()
    comparison = generate_comparison_questions()
    all_questions = bridge + comparison
    random.shuffle(all_questions)

    print(f"Generated {len(bridge)} bridge + {len(comparison)} comparison = {len(all_questions)} total questions")

    # Split
    n_train = min(args.n_train, len(all_questions) - args.n_val)
    n_val = min(args.n_val, len(all_questions) - n_train)
    train = all_questions[:n_train]
    val = all_questions[n_train:n_train + n_val]

    # Count types
    train_bridge = sum(1 for q in train if q["type"] == "bridge")
    train_comp = sum(1 for q in train if q["type"] == "comparison")
    print(f"Train: {len(train)} ({train_bridge} bridge + {train_comp} comparison)")
    print(f"Val: {len(val)}")

    # FoldAgent version
    train_rows = [to_row(q, "search_branch") for q in train]
    val_rows = [to_row(q, "search_branch") for q in val]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/multihop_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/multihop_test.parquet", index=False)
    print(f"Wrote multihop_train.parquet ({len(train_rows)} rows)")

    # ContextGraph version
    train_rows_g = [to_row(q, "search_graph") for q in train]
    val_rows_g = [to_row(q, "search_graph") for q in val]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/multihop_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/multihop_graph_test.parquet", index=False)
    print(f"Wrote multihop_graph_train.parquet ({len(train_rows_g)} rows)")

    # Samples
    print("\nSample questions:")
    for q in train[:8]:
        print(f"  [{q['type']}/{q['hops']}hop] Q: {q['question']}")
        print(f"    A: {q['answer']}  ({q['reasoning']})")


if __name__ == "__main__":
    main()
