import math
import match

from elasticsearch import Elasticsearch
from collections import Counter
from os import linesep

D = 0  # total number of documents in the corpus


def get_top_docs_for_query(es, index, model, qid, top=10):
    doc_ids = []
    with open(f"{match.EVAL_FOLDER}/{model}/l2/ranking.txt", "r") as fp:
        for line in fp:
            if len(doc_ids) >= top:
                break
            [query_id, _, doc_id, rank, score, _] = line.split()
            if int(query_id) == int(qid):
                doc_ids.append(doc_id)
    return [fetch_doc(es, index, _id) for _id in doc_ids]


def fetch_doc(es, index, doc_id):
    raw_doc = es.termvectors(
        index=index,
        id=doc_id,
        fields=["text"],
        term_statistics=True,
        field_statistics=False,
        offsets=False,
        positions=False
    )
    return match.build_doc_struct(raw_doc)


def pseudo_relevance_feedback(es=None, index=None, tokens=None, docs=None, extra=None):
    def find_distinctive_terms(d):
        terms = []
        for term, metadata in d[match.TERMS].items():
            df = metadata["doc_freq"]
            ttf = metadata["ttf"]
            tf = metadata["term_freq"]
            if tf > ttf / df:
                terms.append(term)
        return terms

    common = Counter()
    for doc in docs:
        common.update(find_distinctive_terms(doc))

    interesting_terms = []
    for term, occ in common.most_common(len(tokens) + extra):
        if term not in tokens and len(interesting_terms) < extra:
            interesting_terms.append(term)
    return interesting_terms


def significant_terms(es=None, index=None, tokens=None, docs=None, extra=None):
    common = Counter()
    for token in tokens:
        significant = es.search(index=index, body={
            "query": {
                "terms": {
                    "text": [token]
                }
            },
            "aggregations": {
                "related_words": {
                    "significant_terms": {
                        "field": "text"
                    }
                }
            },
            "size": 0})["aggregations"]["related_words"]["buckets"]
        common.update([term["key"] for term in significant])

    common = Counter({term: occ for term, occ in common.items() if occ > 1 and term not in tokens})
    terms_idf = []
    for term, occ in common.most_common(len(tokens) + extra):
        df = es.count(index=index, body={"query": {"match_phrase_prefix": {"text": term}}})["count"]
        terms_idf.append((term, occ, math.log(D / df)))

    terms_idf.sort(key=lambda term: term[2] + term[1] * 2, reverse=True)
    print(terms_idf)

    return [term for term, _, _ in terms_idf[:extra]]


def improve_query(es=None, index=None, tokens=None, strategy=None, docs=None, extra=3):
    print(f"Original tokens: {tokens}")
    terms = strategy(es, index, tokens, docs, extra)
    print(f"Added terms: {terms}{linesep}")
    return terms


def write_query_file(file_path, queries):
    with open(file_path, "w") as fp:
        for qid, query in queries.items():
            fp.write(f"{qid}.   {query}{linesep}")


def get_metadata(es, index_name):
    global D
    field_stats = es.termvectors(
        index=index_name,
        id="AP890101-0001",
        fields=["text"]
    )["term_vectors"]["text"]["field_statistics"]
    D = field_stats["doc_count"]


def main():
    index = "ap_dataset"
    es = Elasticsearch("http://localhost:9200")
    level2_query_file = "/Users/tianzerun/Desktop/hw1/data/AP_DATA/query/l2.txt"
    level3_query_file = "/Users/tianzerun/Desktop/hw1/data/AP_DATA/query/l3.txt"
    level4_query_file = "/Users/tianzerun/Desktop/hw1/data/AP_DATA/query/l4.txt"

    get_metadata(es, index)
    raw_queries = match.read_queries(level2_query_file)
    tokenized_queries = {key: match.tokenize(es, value) for key, value in raw_queries.items()}

    pr_feedback_queries = dict()
    for qid, tokens in tokenized_queries.items():
        docs = get_top_docs_for_query(es, index, match.M_TF_IDF, qid)
        print(f"Improve query {qid}:")
        added_tokens = improve_query(tokens=tokens, docs=docs, strategy=pseudo_relevance_feedback)
        added_text = " ".join(added_tokens)
        pr_feedback_queries[qid] = raw_queries[qid] + " " + added_text
    write_query_file(level3_query_file, pr_feedback_queries)

    queries_with_significant_terms = dict()
    for qid, tokens in tokenized_queries.items():
        print(f"Improve query {qid}:")
        added_tokens = improve_query(
            es=es,
            index=index,
            tokens=tokens,
            strategy=significant_terms,
            extra=1
        )
        added_text = " ".join(added_tokens)
        queries_with_significant_terms[qid] = raw_queries[qid] + " " + added_text
    write_query_file(level4_query_file, queries_with_significant_terms)


if __name__ == '__main__':
    main()
