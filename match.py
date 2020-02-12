import re
import time
import math
from os import linesep

from heap import MaxHeap

EVAL_FOLDER = "./evaluation"
QUERY_REGEX = re.compile(r"^([0-9]+)\.\s*(.*)")
AVG_DOC_LENGTH = 0
SUM_TTF = 0
D = 0  # total number of documents in corpus
V = 0  # total number of unique terms in corpus
DOC_LENGTH = "length"
DOC_ID = "doc_id"
FIELD_STATS = "field_statistics"
TERMS = "terms"
TF_SCORES = dict()
DF_SCORES = dict()
TTF_SCORES = dict()


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


def query_id_extractor(text):
    match = re.search(QUERY_REGEX, text)
    if match is not None:
        return match.group(1)
    return None


def query_content_extractor(text):
    match = re.search(QUERY_REGEX, text)
    if match is not None:
        return match.group(2)
    return None


def read_queries(file_path):
    queries = dict()
    with open(file_path, "r") as fp:
        for line in fp:
            qid = query_id_extractor(line)
            query_text = query_content_extractor(line)
            if qid is not None and query_text is not None:
                queries[qid] = query_text
    return queries


def retrieve_postings(client, term):
    return client.postings(term)


def build_doc_struct(doc):
    return {
        DOC_ID: doc["_id"],
        TERMS: doc["term_vectors"]["text"][TERMS],
        DOC_LENGTH: sum(stats["term_freq"]
                        for stats in
                        doc["term_vectors"]["text"][TERMS].values())
    }


def get_raw_df(client, term):
    if term not in DF_SCORES:
        df = client.df(term)
        df = D if df == 0 else df
        DF_SCORES[term] = df
    return DF_SCORES[term]


def get_raw_tf(client, doc_id, term):
    return client.tf(term, doc_id)


def get_raw_ttf(client, term):
    if term not in TTF_SCORES:
        TTF_SCORES[term] = client.ttf(term)
    return TTF_SCORES[term]


def okapi_tf(client, doc_id, term):
    if doc_id not in TF_SCORES:
        TF_SCORES[doc_id] = dict()
    if term not in TF_SCORES[doc_id]:
        tf = get_raw_tf(client, doc_id, term)
        TF_SCORES[doc_id][term] = tf / (tf + 0.5 + 1.5 * (client.doc_length(doc_id) / AVG_DOC_LENGTH))
    return TF_SCORES[doc_id][term]


def get_metadata(client):
    global D
    global V
    global AVG_DOC_LENGTH
    global SUM_TTF
    SUM_TTF = client.sum_ttf()
    D = client.doc_count()
    AVG_DOC_LENGTH = SUM_TTF // D
    V = client.unique_word_count()


def get_related_docs(client, tokens):
    relevant_docs = set()
    for token in tokens:
        relevant_docs |= retrieve_postings(client, token)
    return relevant_docs


def write_query_results(qid, ranked_docs, out_fp):
    for rank, (score, doc_id) in enumerate(ranked_docs, start=1):
        out_fp.write(f"{qid} Q0 {doc_id} {rank} {score} Exp{linesep}")


##############################
#           Models           #
##############################
def es_built_in_model(client, tokens):
    res = client.search(body={"query": {"match": {"text": " ".join(tokens)}}},
                        size=1000, _source=False)
    return [(item["_score"], item["_id"]) for item in res["hits"]["hits"]]


def model_template(client, tokens, scoring_func, size=1000):
    doc_ids = get_related_docs(client, tokens)
    client.cache_docs(doc_ids)
    scored_docs = MaxHeap(size)
    scoring_start_time = time.time()
    print(f"    score_docs={len(doc_ids)}(documents)")
    for _id in doc_ids:
        score = scoring_func(client, _id, tokens)
        scored_docs.push((score, _id))
    print(f"    elapsed_t={time_used(scoring_start_time)}")
    return scored_docs.top()


def tf_model(client, d, q):
    return sum(okapi_tf(client, d, w) for w in q)


def tf_idf_model(client, d, q):
    return sum(okapi_tf(client, d, w) * math.log(D / get_raw_df(client, w)) for w in q)


def bm_25_model(client, d, q):
    k1 = 1.5  # k1 is bumped up by 0.2 to the standard 1.2
    k2 = 500
    b = 0.75
    ans = 0
    for w in q:
        tf_w_d = get_raw_tf(client, d, w)
        tf_w_q = q.count(w)
        c1 = math.log((D + 0.5) / (get_raw_df(client, w) + 0.5))
        c2 = (tf_w_d + k1 * tf_w_d) / (tf_w_d + k1 * ((1 - b) + b * (client.doc_length(d) / AVG_DOC_LENGTH)))
        c3 = (tf_w_q + k2 * tf_w_q) / (tf_w_q + k2)
        ans += c1 * c2 * c3
    return ans


def laplace_smoothing_language_model(client, d, q):
    def p_laplace(w, d):
        return (get_raw_tf(client, d, w) + 1) / (client.doc_length(d) + V)

    return sum(math.log(p_laplace(w, d)) for w in q)


def jelinek_mercer_smoothing_language_model(client, d, q):
    def p_jm(w, d):
        # the smoothing parameter lambda
        s_p = 0.8
        score = s_p * (get_raw_tf(client, d, w) / client.doc_length(d)) + (1 - s_p) * (get_raw_ttf(client, w) / SUM_TTF)
        return score if score != 0 else (1 - s_p) * (1 / V)

    return sum(math.log(p_jm(w, d)) for w in q)


def proximity_search_model(client, d, q):
    def _smallest_blurb(positions):
        min_cover = SUM_TTF
        pointers = {term: 0 for term in positions.keys()}
        frontier = {term: positions[0] for term, positions in positions.items()}
        done = set()
        while True:
            min_cover = min(min_cover, max(frontier.values()) - min(frontier.values()))
            reach_end = all(pointer >= len(positions[term]) for term, pointer in pointers.items())
            if reach_end:
                break

            tmp = [(term, pos) for term, pos in frontier.items() if term not in done]
            term_to_be_advanced, _ = min(tmp, key=lambda kv: kv[1])

            pointers[term_to_be_advanced] += 1
            advanced_p = pointers[term_to_be_advanced]
            if advanced_p >= len(positions[term_to_be_advanced]):
                done.add(term_to_be_advanced)
            else:
                frontier[term_to_be_advanced] = positions[term_to_be_advanced][advanced_p]
        return min_cover

    def _bigram_min_cover_score(d, q):
        groups = []
        for i in range(len(q) - 1):
            groups.append((q[i], q[i + 1]))

        score = 0
        for bigram in groups:
            positions_by_term = dict()
            for term in bigram:
                positions = client.positions(term, d)
                if len(positions) > 0:
                    positions_by_term[term] = positions

            if len(positions_by_term) < 2:
                continue

            min_cover = _smallest_blurb(positions_by_term)
            if min_cover == 0:
                score += min_cover
            else:
                normalized_min_cover = min_cover / len(positions_by_term)
                score += math.log(1.5 + math.exp(1 / normalized_min_cover))

        return score

    def _min_dist_score(d, q):
        positions_by_term = dict()
        for term in q:
            positions = client.positions(term, d)
            if len(positions) > 0:
                positions_by_term[term] = positions

        term_in_both_dq = list(positions_by_term.keys())
        pairs = []
        for i in range(len(term_in_both_dq) - 1):
            for j in range(i + 1, len(term_in_both_dq)):
                pairs.append((term_in_both_dq[i], term_in_both_dq[j]))

        if len(pairs) == 0:
            min_dist = client.doc_length(d)
        else:
            min_dist = min(_smallest_blurb({term: positions_by_term[term] for term in pair}) for pair in pairs)

        return math.log(1.5 + math.exp(0 - min_dist))

    return _bigram_min_cover_score(d, q) + tf_idf_model(client, d, q)


##############################
#       Model Runner         #
##############################
def run_engine_with(model, client, queries, out_fp):
    print(f"Model: {model.__name__}")
    run_model_start_time = time.time()
    for qid, tokens in queries.items():
        start = time.time()
        print(f"Run query={qid} with {model.__name__}")
        print(f"    tokens={tokens}")
        top_docs = model_template(client=client, tokens=tokens, scoring_func=model)
        write_query_results(qid, top_docs, out_fp)
        print(f"    total_time={time_used(start)}")
    print(f"Elapsed time: {time_used(run_model_start_time)}{linesep * 3}")
