import os
import copy
import collections
import difflib
import numpy as np
import re, unicodedata
from collections import Counter
import ast
import asyncio, json, httpx
import logging
import string
if __package__:
    from .judge_client import call_openai_raw
else:
    from judge_client import call_openai_raw

# call this once early (after your logging.basicConfig if you use it)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", judge_response,
                             re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", judge_response,
                                 re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning/explanation
    reasoning_match = re.search(r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                    judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract correct (yes/no)
    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    # Extract confidence (percentage)
    confidence_match = re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    # Check if we got the essential fields
    if result["correct"] is None:
        result["parse_error"] = True

    return result


def extract_citations_from_response(response_text: str):
    """Extract citations from response text
      - [docid] or [docid1, docid2, ...]
      - 【docid】 or 【docid1, docid2, ...】 (oss was finetuned on this format)
    """
    if not response_text:
        return []

    # [docid]
    single_citation_pattern = r'\[(\d+)\]'
    single_matches = re.findall(single_citation_pattern, response_text)

    multi_citation_pattern = r'\[([^\[\]]*?)\]'
    multi_matches = re.findall(multi_citation_pattern, response_text)

    # 【docid】
    single_fullwidth_pattern = r'【(\d+)】'
    single_fullwidth_matches = re.findall(single_fullwidth_pattern, response_text)

    multi_fullwidth_pattern = r'【([^【】]*?)】'
    multi_fullwidth_matches = re.findall(multi_fullwidth_pattern, response_text)

    all_docids = set()

    all_docids.update(single_matches)
    all_docids.update(single_fullwidth_matches)

    for match in multi_matches:
        if match in single_matches:
            continue
        docids = re.findall(r'\d+', match)
        all_docids.update(docids)

    for match in multi_fullwidth_matches:
        if match in single_fullwidth_matches:
            continue
        docids = re.findall(r'\d+', match)
        all_docids.update(docids)

    return list(all_docids)


def em_score(label: str, pred: str) -> bool:
    ign = {'a', 'an', 'the', 'of', 'on', 'in', 'and', '&', 'for', 'to', 'by', 'with'}
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    def norm(s: str) -> str:
        s = deacc(s).lower()
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)  # drop parenthetical qualifiers: (Egypt), (US), etc.
        s = re.sub(r'[“”"\'`]+', '', s)  # drop quotes
        s = re.sub(r'[:–—\-_/.,;!()?]+', ' ', s)  # unify punctuation to spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    strip = lambda s: re.sub(r'\s+', '', norm(s))
    toks = lambda s: [t for t in norm(s).split() if t not in ign and not re.fullmatch(r'\d{4}', t)]
    if strip(label) == strip(pred): return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt: return False
    if Counter(lt) == Counter(pt): return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))): return True
    head = lambda s: strip(re.split(r'[:–—-]', norm(s), 1)[0])
    if head(label) == head(pred): return True
    return False

def relaxed_em(label: str, pred: str) -> bool:
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    norm  = lambda s: re.sub(r'\s+',' ',re.sub(r'\s*\([^)]*\)\s*',' ',re.sub(r'[“”"\'`]+','',re.sub(r'[:–—\-_/.,;!()?]+',' ',deacc(s).lower())))).strip()
    strip = lambda s: re.sub(r'\s+','',norm(s))
    if not label or not pred: return False
    A,B = strip(label), strip(pred)
    if A==B or A in B or B in A: return True
    if difflib.SequenceMatcher(None,A,B).ratio()>=0.9: return True
    ca,cb=Counter(A),Counter(B);
    if sum((ca&cb).values())/min(len(A),len(B) or 1)>=0.9: return True
    return False


def searchr1_normalize_answer(value: str) -> str:
    """Match Search-R1's QA exact-match normalization."""
    value = str(value).lower()
    punctuation = set(string.punctuation)
    value = "".join(character for character in value if character not in punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def searchr1_em_score(labels, prediction: str) -> bool:
    if isinstance(labels, str):
        labels = [labels]
    normalized_prediction = searchr1_normalize_answer(prediction)
    return any(searchr1_normalize_answer(label) == normalized_prediction for label in labels)


async def judge(question, correct_answer, predicted_answer, audit_sink=None, *, raise_errors=False):
    # Patch browsecomp typo
    correct_answer = "ttellomS saiboT"[::-1] if "tellomS saiboT"[::-1] in correct_answer else correct_answer  # fix
    correct_answer = "yayhdapottahC najnarawsiB"[::-1] if "yayhdapattahC najnarawsiB"[::-1] in correct_answer else correct_answer
    predicted_answer = "yrtnuoC a fo htaP ehT :sedirelC sokfalG"[::-1] if "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1] in predicted_answer else predicted_answer
    strict_match = em_score(correct_answer, predicted_answer)
    relaxed_match = relaxed_em(correct_answer, predicted_answer)
    audit = {
        "question": question,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "strict_em": strict_match,
        # Diagnostic only. This heuristic is intentionally not allowed to
        # override a strict-EM miss or a negative/failed LLM judgment.
        "relaxed_em": relaxed_match,
        "judge_model": None,
        "judge_method": None,
        "grader_attempts": [],
        "score": 0,
    }
    if strict_match:
        score = 1
        audit["judge_method"] = "strict_em"
    elif len(predicted_answer.strip()) == 0:
        score = 0
        audit["judge_method"] = "blank_prediction"
    else:
        # Skip the OpenAI grader if no real API key is set. A strict-EM miss
        # stays negative in offline mode: relaxed_em is too permissive to be a
        # safe source of positive outcome labels for policy training.
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "dummy":
            score = 0
            audit["judge_method"] = "offline_strict_only"
        else:
            judge_prompt = GRADER_TEMPLATE.format(
                question=question,
                response=predicted_answer,
                correct_answer=correct_answer
            )
            messages = [{'role': 'user', 'content': judge_prompt}]
            judge_model = os.getenv("JUDGE_MODEL", "gpt-5-nano")
            audit["judge_model"] = judge_model
            score = 0
            audit["judge_method"] = "llm_parse_failure"
            for attempt in range(3):
                if raise_errors:
                    response = await call_openai_raw(messages, model=judge_model, raise_errors=True)
                else:
                    response = await call_openai_raw(messages, model=judge_model)
                grade_report = parse_judge_response(response)
                audit["grader_attempts"].append({
                    "attempt": attempt + 1,
                    "response": response,
                    "parsed": grade_report,
                })
                if grade_report['parse_error']:
                    continue
                # Defensive: parse_judge_response can return correct=None even
                # when parse_error is False; treat None as 0.
                score = int(grade_report.get('correct') or 0)
                audit["judge_method"] = "llm_judge"
                break

    audit["score"] = score
    if audit_sink is not None:
        audit_sink.append(copy.deepcopy(audit))
    print("[JUDGE AUDIT] " + json.dumps({
        "score": score,
        "method": audit["judge_method"],
        "strict_em": strict_match,
        "relaxed_em": relaxed_match,
        "judge_model": audit["judge_model"],
    }, ensure_ascii=False))
    print(f"[Judged] score={score}\nLabel: " + correct_answer + '\nModel: ' + predicted_answer.split('\n')[0])
    return score


def keep_first_n_words(text: str, n: int = 1000, max_chars: int | None = None) -> str:
    if not text:
        return ""
    result = text
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == n:
            result = text[:m.end()] + '\n[Document is truncated.]'
            break
    if max_chars is not None and max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars].rstrip() + '\n[Document is truncated.]'
    return result


class AsyncSearchClient:
    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3, backoff: float = 0.5):
        self.base_urls = [url.strip().rstrip("/") for url in base_url.split(",") if url.strip()]
        if not self.base_urls:
            raise ValueError("at least one local search base URL is required")
        self.base_url = self.base_urls[0]
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._clients = [httpx.AsyncClient(base_url=url) for url in self.base_urls]
        self._next_client = 0
        self._close_task = None

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_all())
        await asyncio.shield(self._close_task)

    async def _close_all(self):
        results = await asyncio.gather(
            *(client.aclose() for client in self._clients), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _round_robin_client(self):
        client = self._clients[self._next_client]
        self._next_client = (self._next_client + 1) % len(self._clients)
        return client

    async def _post(self, path: str, payload: dict):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                client = self._round_robin_client()
                r = await client.post(path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("results", data)  # convenience: unwrap "results" if present
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == self.retries:
                    raise
                await asyncio.sleep(self.backoff * attempt)
        raise last_exc  # should not reach

    async def search(self, query: str, k: int = 10):
        return await self._post("/search", {"query": query, "k": k})

    async def open(self, url: str | None = None, docid: str | None = None):
        return await self._post("/open", {"url": url, "docid": docid})


def extract_json_tool(text: str):
    """Parse native Search-R1 tags and JSON-style tool blocks."""
    calls = []
    def parse_obj(s):
        for p in (json.loads, ast.literal_eval):
            try: return p(s)
            except Exception: pass
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            frag = m.group(0)
            for p in (json.loads, ast.literal_eval):
                try: return p(frag)
                except Exception: pass
        return None
    native_action = re.search(r"<(answer|search)>\s*(.*?)\s*</\1>", text, flags=re.S)
    if native_action:
        kind, body = native_action.groups()
        if kind == "answer":
            return [{"function": "finish", "arguments": {"answer": body.strip()}}]
        return [{"function": "search", "arguments": {"query": body.strip(), "topk": 3}}]

    for kind, body in re.findall(r"<(tool_call)>\s*(.*?)\s*</\1>", text, flags=re.S):
        body = body.strip()
        if kind == "tool_call":
            if body.startswith("```") and body.endswith("```"):
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
            obj = parse_obj(body)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                calls.append({"function": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    aligned_calls = []
    for fn in calls:
        if fn['function'] == "search":
            queries = fn['arguments'].get('query', [])
            if isinstance(queries, str):
                queries = [queries]
            topk = max(10 // (len(queries) + 1), 2)
            for q in queries:
                aligned_calls.append({"function": "search", "arguments": {"query": q, "topk": topk}})
        elif fn['function'] == "visit":
            for url in fn['arguments'].get('url', []):
                aligned_calls.append({"function": "open_page", "arguments": {"url": url}})
        else:
            aligned_calls.append(fn)
    return aligned_calls

def extract_fn_call(text):
    if not text:
        return None
    if '<tool_call>' in text or '<answer>' in text or '<search>' in text:
        json_tool = extract_json_tool(text)
        if len(json_tool) > 0:
            return json_tool
        else:
            print(text)
    text = re.split(r'<\[[^\]]+\]>', text)[-1].strip()
    matches = list(re.finditer(r'(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>',
                               text, re.DOTALL))
    if not matches:
        return None
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count('\n', prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]
    return [
        {
            'function': m.group(1),  # <-- each call uses its *own* captured fn name
            'arguments': dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',
                                         m.group(2), re.DOTALL))
        }
        for m in last
    ]


class LocalSearch:
    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.stats['finish'] = 0
        self.stats['search'] = 0
        self.stats['open_page'] = 0
        self.stats['change_answer'] = 0
        self.stats['is_search'] = 0
        self.stats['is_open'] = 0
        self.stats['is_finish'] = 0
        self.stats['visit_pages'] = 0
        self.env_fail = False

        base_url = os.getenv("LOCAL_SEARCH_URL")

        self.question = None
        self.label_answer = None
        self.predicted_answer = None
        self.judge_audit = []
        self.double_check = getattr(self.config.plugin, "double_check", False)
        self.donotgiveup = False
        self.must_search = getattr(self.config.plugin, "must_search", True)
        self.use_skills_only_memory = bool(
            getattr(self.config.plugin, "use_skills_only_memory", False)
        )
        self.stats['skill_bank_enabled'] = int(self.use_skills_only_memory)
        self.stats['skill_bank_injected'] = 0
        self.skills_json_path = str(
            getattr(self.config.plugin, "skills_json_path", "")
        ).strip()
        self.skills_top_k = max(0, int(getattr(self.config.plugin, "skills_top_k", 6)))
        if self.use_skills_only_memory and not self.skills_json_path:
            raise ValueError("plugin.skills_json_path is required when static SkillBank is enabled")
        self.search_topk_cap = max(1, int(getattr(self.config.plugin, "search_topk_cap", 10)))
        self.search_snippet_words = max(1, int(getattr(self.config.plugin, "search_snippet_words", 512)))
        self.search_snippet_chars = max(128, int(getattr(self.config.plugin, "search_snippet_chars", 12000)))
        self.open_page_words = max(1, int(getattr(self.config.plugin, "open_page_words", 4096)))
        self.open_page_chars = max(256, int(getattr(self.config.plugin, "open_page_chars", 48000)))
        self.visited_pages = set()
        # Trusted tool-returned text only; never populated from model citations.
        self.evidence_documents = []
        self.record_evidence = (getattr(self.config.plugin, 'graph_rpo_credit_backend', '') == 'evidence'
                                or getattr(self.config.plugin, 'graph_branch_history', False))
        self.is_finish = False
        self.emergency_finish_wrapped = False
        # Allocate connections only after all constructor validation succeeds.
        self.client = AsyncSearchClient(base_url=base_url)

    async def init_env(self, item):
        extra = item.non_tensor_batch['extra_info']
        extra = extra.item() if hasattr(extra, 'ndim') and extra.ndim == 0 else extra[0]
        self.question = extra['query']
        self.label_answer = extra['answer']
        aliases = extra.get('answer_aliases', [self.label_answer])
        if hasattr(aliases, 'tolist'):
            aliases = aliases.tolist()
        if isinstance(aliases, str):
            aliases = [aliases]
        self.answer_aliases = [str(answer) for answer in aliases if str(answer).strip()]
        if not self.answer_aliases:
            self.answer_aliases = [self.label_answer]
        self.reward_mode = extra.get('reward_mode', 'default')
        self.predicted_answer = None
        self.judge_audit = []
        self.instance_info = copy.deepcopy(extra)
        self.evidence_documents = []
        self.instance_info['problem_statement'] = self.instance_info['query']
        self.search_skill_context = None
        self.search_skill_context_injected = False
        if self.use_skills_only_memory:
            from envs.search_skill_bank import format_search_skills

            self.search_skill_context = format_search_skills(
                self.question,
                self.skills_json_path,
                self.skills_top_k,
            )

    async def run_action(self, response):
        self.stats['action'] += 1
        fn_call = extract_fn_call(response)
        if fn_call is None or len(fn_call) == 0:
            # Improved message for no function call
            return {'observation': 'No function call was detected in the model response.'}
        else:
            observation = ''
            for fn in fn_call:
                name = fn['function']
                if name == 'search':
                    self.stats['search'] += 1
                    query = fn['arguments'].get('query', '')
                    topk = (lambda v: int(v) if str(v).isdigit() else 10)(fn['arguments'].get('topk', 10))
                    topk = min(max(topk, 1), self.search_topk_cap)
                    if not query:
                        observation += '[Error] The "search" function requires a "query" argument.'
                    else:
                        observation += f'[Search Results for "{query}"]\n'
                        serp = await self.client.search(query, 50)
                        self.stats['is_search'] = 1
                        show_topk = 0
                        for i, page in enumerate(serp, 1):
                            # Formatted entry for each search result
                            if page['docid'] in self.visited_pages:
                                page['text'] = "(This page was already seen in a previous search. Here, a shorter snippet is shown. If you find this page relevant, please use the open_page tool to inspect the full content) " + keep_first_n_words(page['text'], 128, self.search_snippet_chars)
                                show_topk += 0.25
                            else:
                                self.visited_pages.add(page['docid'])
                                self.stats['visit_pages'] = len(self.visited_pages)
                                page['text'] = keep_first_n_words(
                                    page['text'], self.search_snippet_words,
                                    self.search_snippet_chars,
                                )
                                show_topk += 1
                            if self.record_evidence:
                                self.evidence_documents.append({'docid': str(page['docid']), 'text': page['text']})
                            observation += (
                                f"\n--- #{i}: {page['docid']}---\n"
                                f"docid: {page['docid']}\n"
                                f"url: {page['url']}\n"
                                f"content: {page['text']}\n"
                            )
                            if show_topk >= topk:
                                break
                        observation += "\n"

                elif name == 'open_page':
                    self.stats['open_page'] += 1
                    self.stats['is_open'] = 1
                    url = fn['arguments'].get('url', None)
                    docid = fn['arguments'].get('docid', None)
                    if not docid and not url:
                        # Clearer error for missing parameters
                        observation += '[Error] The "open_page" function requires either a "docid" or a "url".'
                    else:
                        open_pages = await self.client.open(url, docid)
                        for page in open_pages:
                            # Structured format for opened page content
                            page['text'] = keep_first_n_words(
                                page['text'], self.open_page_words,
                                self.open_page_chars,
                            )
                            if self.record_evidence and page.get('docid') is not None and page['text'] not in ('Document not found for given docid.', 'Missing docid and url, or url not indexed.'):
                                self.evidence_documents.append({'docid': str(page['docid']), 'text': page['text']})
                            observation += (
                                f"[Opened Page Content]\n"
                                f"docid: {page['docid']}\n"
                                f"url: {page['url']}\n"
                                f"content: {page['text']}\n"
                            )
                        observation += "\n"
                elif name == 'finish':
                    answer = fn['arguments'].get('answer', "")
                    explanation = fn['arguments'].get('explanation', None)
                    confidence = fn['arguments'].get('confidence', None)
                    if len(answer.strip()) == 0:
                        print(response)
                        observation = ("Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
                                       "<function=finish>\n" "<parameter=answer>YOUR ANSWER</parameter>\n"
                                       "<parameter=explanation>YOUR EXPLANATION</parameter>\n"
                                       "<parameter=confidence>YOUR CONFIDENCE</parameter>\n" "</function>\n")
                        return {'observation': observation.strip()}
                    if self.predicted_answer is not None:
                        if self.predicted_answer[0] != answer:
                            self.stats['change_answer'] += 1

                    if not self.stats['is_search']:
                        if self.must_search:
                            observation = "Answer submission failed. You MUST use the search tool to verify the answer and all the evidence, and cite the correct source document in your explanation to support your claim."
                            return {'observation': observation.strip()}
                        answer = ""  # No search no reward

                    if 'insufficient' in answer.lower() and self.donotgiveup:
                        observation = "The answer is guaranteed to be found through sufficient search and reading. Do not give up; try searching deeper or using alternative approaches."
                        self.donotgiveup = False
                        return {'observation': observation.strip()}

                    if '<q1>' in self.label_answer:
                        label_answer_dict = extract_q_dict(self.label_answer)
                        predicted_answer_dict = extract_q_dict(answer)
                        missing = []
                        for k in label_answer_dict:
                            if k not in predicted_answer_dict:
                                missing.append(k)

                        if len(missing) > 0:
                            observation = f"Answer submission failed. The answer is missing the following questions: {', '.join(missing)}. Make sure submit answer for all the questions. Ensure all the answers are submitted in one finish tool call."
                            return {'observation': observation.strip()}

                    if self.double_check:  # disabled
                        observation = f"""Before finalizing, perform this mandatory check.

Check Against the Goal: Reread the user's original query: {self.question}. 

Does your answer perfectly and completely satisfy every single condition? Create a checklist of all conditions from the query, and verify them one by one. Ensure every item on the checklist is satisfied.

Verify the Evidence: For each fact in your answer, confirm it is explicitly supported by the source documents you read. Inference is not permitted. The evidence must be direct.

Take Corrective Action: If you notice any gaps or unsupported points, revisit the sources and refine your answer. If there is any unverified claim or unmet condition, return to research to find the correct information and construct a fully verified answer. 

Once you’re confident everything is covered and verified, submit the final answer and include enough citations for all supporting evidence."""
                        self.double_check = False
                        return {'observation': observation.strip()}
                    self.predicted_answer = (answer, explanation, confidence)
                    self.stats['is_finish'] = 1
                    self.is_finish = True
                    return {'action': 'finish'}
                else:
                    # Clearer error for unsupported functions
                    observation = f'[Error] The function "{name}" is not supported.'
            if (
                observation
                and self.search_skill_context
                and not self.search_skill_context_injected
            ):
                observation += f"\n\n{self.search_skill_context}"
                self.search_skill_context_injected = True
                self.stats['skill_bank_injected'] = 1
            observation += "\n\n* Please reflect on the information we have obtained, and keep searching for additional information if we still can not answer the question. Do not give the answer if the information is still not enough."

        return {'observation': observation.strip()}

    async def score_answer(self, predicted_answer, audit_sink=None):
        """Score an arbitrary answer without mutating the episode state."""
        if getattr(self, 'reward_mode', 'default') == 'searchr1_em':
            score = int(searchr1_em_score(self.answer_aliases, predicted_answer))
            audit = {
                "question": self.question,
                "correct_answers": self.answer_aliases,
                "predicted_answer": predicted_answer,
                "strict_em": bool(score),
                "judge_model": None,
                "judge_method": "searchr1_em",
                "score": score,
            }
            if audit_sink is not None:
                audit_sink.append(copy.deepcopy(audit))
            print("[JUDGE AUDIT] " + json.dumps({
                "score": score,
                "method": "searchr1_em",
                "strict_em": bool(score),
                "judge_model": None,
            }, ensure_ascii=False))
            print(f"[Judged] score={score}\nLabel: {self.answer_aliases[0]}\nModel: " + predicted_answer.split('\n')[0])
            return score
        judge_kwargs = {"raise_errors": True} if getattr(self, "raise_judge_errors", False) else {}
        if '<q1>' in self.label_answer:
            label_answer_dict = extract_q_dict(self.label_answer)
            predicted_answer_dict = extract_q_dict(predicted_answer)
            all_reward = []
            for key, label in label_answer_dict.items():
                if key in predicted_answer_dict:
                    reward = await judge(
                        self.question,
                        label,
                        predicted_answer_dict[key],
                        audit_sink=audit_sink,
                        **judge_kwargs,
                    )
                    all_reward.append(reward)
                else:
                    all_reward.append(0)
            return sum(all_reward) / len(all_reward)
        return await judge(
            self.question,
            self.label_answer,
            predicted_answer,
            audit_sink=audit_sink,
            **judge_kwargs,
        )

    async def aclose(self):
        await self.client.close()

    async def get_reward(self, item, messages, context):
        try:
            return await self._get_reward(item, messages, context)
        finally:
            await self.client.close()

    async def _get_reward(self, item, messages, context):
        if self.env_fail:  # If env fail, direct return 0 reward
            return "", 0, {}
        if not self.is_finish or self.predicted_answer is None:
            # A search query, rejected finish, or unsubmitted guess is not an
            # answer. Only the environment's accepted finish can earn reward.
            return "", 0, {}
        reward = await self.score_answer(
            self.predicted_answer[0], audit_sink=self.judge_audit
        )
        self._record_judge_stats()
        return "", reward, {}

    def _record_judge_stats(self):
        audits = self.judge_audit
        self.stats["judge_calls"] = len(audits)
        self.stats["judge_positive"] = sum(int(audit.get("score", 0) > 0) for audit in audits)
        self.stats["judge_strict_em"] = sum(bool(audit.get("strict_em")) for audit in audits)
        self.stats["judge_relaxed_em"] = sum(bool(audit.get("relaxed_em")) for audit in audits)
        self.stats["judge_relaxed_only"] = sum(
            bool(audit.get("relaxed_em")) and not bool(audit.get("strict_em"))
            for audit in audits
        )
        self.stats["judge_llm"] = sum(audit.get("judge_method") == "llm_judge" for audit in audits)
        self.stats["judge_parse_failure"] = sum(
            audit.get("judge_method") == "llm_parse_failure" for audit in audits
        )

    async def update_dataproto(self, out, item, messages, score, reward_dict, tag='main', metrics=None):
        final_score = score[1]
        out.meta_info["inference_metrics"] = metrics
        out.meta_info["generation_kwargs"] = item.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        out.non_tensor_batch["num_of_turns"] = np.array([len(messages)], dtype=object)
        out.non_tensor_batch["turn_clipped"] = np.array([False], dtype=object)
        out.non_tensor_batch["tag"] = np.array([tag, ], dtype=object)
        out.non_tensor_batch["is_summary"] = np.array([int("summary" in tag), ], dtype=object)
        out.non_tensor_batch["traj_cnt"] = np.array([1, ], dtype=object)
        stats = dict(self.stats)
        stats['score'] = final_score
        extra_data = {"score": score, "call_fail": self.env_fail, "action_fail": 0, "answer_reached": True,
                      "stats": stats}
        out.non_tensor_batch['extra_data'] = np.array([extra_data, ], dtype=object)
        return out

def extract_q_dict(s: str) -> dict[str, str]:
    return {k: v.strip() for k, v in re.findall(r'<(q\d+)>(.*?)</\1>', s, flags=re.S)}


def _snippet(page: dict, n: int = 300) -> str:
    txt = page.get("text") or page.get("raw") or page.get("content") or ""
    return txt[:n].replace("\n", " ")


async def _run_simple_test():
    base_url = os.getenv("LOCAL_SEARCH_URL")
    client = AsyncSearchClient(base_url=base_url)
    try:
        query = "Elon Musk"  # change if you like
        print(f"\n=== SEARCH: {query!r} ===")
        serp = await client.search(query, k=3)
        print(f"got {len(serp)} results")
        for i, r in enumerate(serp, 1):
            print(f"{i}. docid={r.get('docid')}  url={r.get('url')}")

        if not serp:
            return

        first = serp[0]
        print("\n=== OPEN by docid ===")
        opened_by_docid = await client.open(docid=first.get("docid"))
        if opened_by_docid:
            print(f"docid={opened_by_docid[0].get('docid')}\nurl={opened_by_docid[0].get('url')}\n"
                  f"content: {_snippet(opened_by_docid[0])}")

        print("\n=== OPEN by url ===")
        opened_by_url = await client.open(url=first.get("url"))
        if opened_by_url:
            print(f"docid={opened_by_url[0].get('docid')}\nurl={opened_by_url[0].get('url')}\n"
                  f"content: {_snippet(opened_by_url[0])}")
    finally:
        await client.close()
