import re
import sys
import time
import math
from os import linesep
from elasticsearch import Elasticsearch, helpers
from heap import MaxHeap

RANKING_FOLDER = "/Users/tianzerun/Desktop/hw1/code/evaluation"
QUERY_REGEX = re.compile(r"^([0-9]+)\.\s*(.*)")
AVG_DOC_LENGTH = 0
SUM_TTF = 0
D = 0  # total number of documents in corpus
V = 232575  # total number of unique terms in corpus (229252 counted by this program)
DOC_LENGTH = "length"
DOC_ID = "doc_id"
FIELD_STATS = "field_statistics"
TERMS = "terms"
DOCS = dict()
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


def tokenize(es, index, text):
    res = es.indices.analyze(
        index=index,
        body={
            "analyzer": "stopped",
            "text": text
        }
    )
    return [token["token"] for token in res.get("tokens", [])]


def retrieve_postings(es, term):
    res = helpers.scan(
        client=es,
        query={"query": {"match": {"text": term}}},
        _source=False
    )
    return set(doc["_id"] for doc in res)


def build_doc_struct(doc):
    return {
        DOC_ID: doc["_id"],
        TERMS: doc["term_vectors"]["text"][TERMS],
        DOC_LENGTH: sum(stats["term_freq"]
                        for stats in
                        doc["term_vectors"]["text"][TERMS].values())
    }


def fetch_docs(es, index_name, ids):
    ids = list(set(ids) - set(DOCS.keys()))
    fetching_docs_start_time = time.time()
    size = 50
    p = 0
    while p < len(ids):
        res = es.mtermvectors(
            index=index_name,
            fields=["text"],
            ids=ids[p:p + size],
            term_statistics=False,
            field_statistics=False,
            offsets=False,
            positions=False
        )
        for doc in res["docs"]:
            doc = build_doc_struct(doc)
            DOCS[doc[DOC_ID]] = doc
        p += size
    print(f"    num_of_docs_fetched={len(ids)}")
    print(f"    fetching_docs_takes={time_used(fetching_docs_start_time)}")


def get_raw_df(es, index, term):
    if term not in DF_SCORES:
        DF_SCORES[term] = es.count(
            index=index,
            body={"query": {"term": {"text": term}}}
        ).get("count", 0)
    return DF_SCORES[term]


def get_raw_tf(doc, term):
    return 0 if doc[TERMS].get(term) is None else doc[TERMS][term]["term_freq"]


def get_taw_ttf(es, index, term):
    if term not in TTF_SCORES:
        doc_id = es.search(
            index=index,
            body={"query": {"match": {"text": term}}},
            _source=False, size=1
        )["hits"]["hits"][0]["_id"]
        TTF_SCORES[term] = es.termvectors(
            index=index,
            id=doc_id,
            fields=["text"],
            body={
                "filter": {
                    "min_word_length": len(term),
                    "max_word_length": len(term)
                }
            },
            term_statistics=True,
            field_statistics=False,
            offsets=False,
            positions=False
        )["term_vectors"]["text"]["terms"][term]["ttf"]
    return TTF_SCORES[term]


def okapi_tf(doc, term):
    doc_id = doc[DOC_ID]
    if doc_id not in TF_SCORES:
        TF_SCORES[doc_id] = dict()
    if term not in TF_SCORES[doc_id]:
        tf = get_raw_tf(doc, term)
        TF_SCORES[doc_id][term] = tf / (tf + 0.5 + 1.5 * (doc[DOC_LENGTH] / AVG_DOC_LENGTH))
    return TF_SCORES[doc_id][term]


def get_metadata(es, index_name):
    global D
    global AVG_DOC_LENGTH
    global SUM_TTF
    field_stats = es.termvectors(
        index=index_name,
        id="AP890101-0001",
        fields=["text"]
    )["term_vectors"]["text"][FIELD_STATS]
    SUM_TTF = field_stats["sum_ttf"]
    D = field_stats["doc_count"]
    AVG_DOC_LENGTH = SUM_TTF // D


def get_related_docs(es, tokens):
    relevant_docs = set()
    for token in tokens:
        relevant_docs |= retrieve_postings(es, token)
    return relevant_docs


def write_query_results(qid, ranked_docs, result_folder):
    with open(f"{result_folder}/ranking.txt", "a") as fp:
        for rank, (score, doc_id) in enumerate(ranked_docs, start=1):
            fp.write(f"{qid} Q0 {doc_id} {rank} {score} Exp{linesep}")


##############################
#           Models           #
##############################
def es_built_in_model(es, index, tokens):
    res = es.search(index=index,
                    body={"query": {"match": {"text": " ".join(tokens)}}},
                    size=1000, _source=False)
    return [(item["_score"], item["_id"]) for item in res["hits"]["hits"]]


def model_template(es, index, tokens, scoring_func, size=1000):
    doc_ids = get_related_docs(es, tokens)
    fetch_docs(es, index, list(doc_ids))
    scored_docs = MaxHeap(size)
    scoring_start_time = time.time()
    for _id in doc_ids:
        doc = DOCS[_id]
        score = scoring_func(doc, tokens)
        scored_docs.push((score, _id))
    print(f"    num_of_docs_analyzed={len(doc_ids)}")
    print(f"    scoring_docs_takes={time_used(scoring_start_time)}")
    return scored_docs.top()


def tf_model(es, index, tokens):
    return model_template(
        es=es,
        index=index,
        tokens=tokens,
        scoring_func=lambda d, q: sum(okapi_tf(d, w) for w in q)
    )


def tf_idf_model(es, index, tokens):
    return model_template(
        es=es,
        index=index,
        tokens=tokens,
        scoring_func=lambda d, q: sum(okapi_tf(d, w) * math.log(D / get_raw_df(es, index, w)) for w in q)
    )


def bm_25_model(es, index, tokens):
    def score(d, q):
        k1 = 1.2
        k2 = 500
        b = 0.75
        ans = 0
        for w in q:
            tf_w_d = get_raw_tf(d, w)
            tf_w_q = q.count(w)
            c1 = math.log((D + 0.5) / (get_raw_df(es, index, w) + 0.5))
            c2 = (tf_w_d + k1 * tf_w_d) / (tf_w_d + k1 * ((1 - b) + b * (d[DOC_LENGTH] / AVG_DOC_LENGTH)))
            c3 = (tf_w_q + k2 * tf_w_q) / (tf_w_q + k2)
            ans += c1 * c2 * c3
        return ans

    return model_template(
        es=es,
        index=index,
        tokens=tokens,
        scoring_func=score
    )


def laplace_smoothing_language_model(es, index, tokens):
    def p_laplace(w, d):
        return (get_raw_tf(d, w) + 1) / (d[DOC_LENGTH] + V)

    return model_template(
        es=es,
        index=index,
        tokens=tokens,
        scoring_func=lambda d, q: sum(math.log(p_laplace(w, d)) for w in q)
    )


def jelinek_mercer_smoothing_language_model(es, index, tokens):
    def p_jm(w, d):
        # the smoothing parameter lambda
        s_p = 0.8
        return s_p * (get_raw_tf(d, w) / d[DOC_LENGTH]) + (1 - s_p) * (get_taw_ttf(es, index, w) / SUM_TTF)

    return model_template(
        es=es,
        index=index,
        tokens=tokens,
        scoring_func=lambda d, q: sum(math.log(p_jm(w, d)) for w in q)
    )


##############################
#       Model Runner         #
##############################
def run_engine_with(model, es, index, queries, result_file):
    print(f"Model: {model.__name__}")
    run_model_start_time = time.time()
    for qid, tokens in queries.items():
        start = time.time()
        print(f"Run query={qid} with {model.__name__}")
        print(f"    tokens={tokens}")
        write_query_results(qid, model(es=es, index=index, tokens=tokens),
                            result_file)
        print(f"    total_time={time_used(start)}")
    print(f"Elapsed time: {time_used(run_model_start_time)}{linesep * 3}")


##############################
#            Main            #
##############################
# Available models to use for the purpose of scoring documents given a query.
# Do not change the representations arbitrarily as they map to dir names.
M_BM_25 = "BM-25"
M_ES_BUILT_IN = "ES-BUILT-IN"
M_TF = "TF"
M_TF_IDF = "TF-IDF"
M_ULM_LAPLACE = "ULM-LAPLACE"
M_ULM_JM = "ULM-JM"


def main(index_name, file_path, *models):
    model_mapping = {
        M_ES_BUILT_IN: es_built_in_model,
        M_TF: tf_model,
        M_TF_IDF: tf_idf_model,
        M_BM_25: bm_25_model,
        M_ULM_LAPLACE: laplace_smoothing_language_model,
        M_ULM_JM: jelinek_mercer_smoothing_language_model
    }
    es = Elasticsearch("http://localhost:9200")
    get_metadata(es, index_name)
    tokenized_queries = {key: tokenize(es, index_name, value)
                         for key, value in read_queries(file_path).items()}
    for model in models:
        run_engine_with(
            model=model_mapping[model],
            es=es,
            index=index_name,
            queries=tokenized_queries,
            result_file=f"{RANKING_FOLDER}/{model}"
        )


if __name__ == '__main__':
    engine_start_time = time.time()
    query_file_path = "/Users/tianzerun/Desktop/hw1/data/AP_DATA/simplified_query_l2.txt"

    use_models = (M_ES_BUILT_IN, M_TF, M_TF_IDF, M_BM_25, M_ULM_LAPLACE, M_ULM_JM)
    main("ap_dataset", query_file_path, M_BM_25)
    print(TTF_SCORES)

    print(f"Total time: {time_used(engine_start_time)}{linesep}")
    sys.exit()
