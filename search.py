import time
from abc import ABC, abstractmethod
from elasticsearch import Elasticsearch, helpers


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


class Search(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def postings(self, term):
        pass

    @abstractmethod
    def df(self, term):
        pass

    @abstractmethod
    def tf(self, term, doc):
        pass

    @abstractmethod
    def ttf(self, term):
        pass

    @abstractmethod
    def doc_count(self):
        pass

    @abstractmethod
    def unique_word_count(self):
        pass

    @abstractmethod
    def sum_ttf(self):
        pass

    @abstractmethod
    def doc_length(self, doc_id):
        pass


class ESearch(Search):
    def __init__(self, hostname, index, field):
        self._es = Elasticsearch(hostname)
        self._index = index
        self._field = field
        self._docs = dict()

    def tokenize(self, text):
        res = self._es.indices.analyze(
            index=self._index,
            body={
                "analyzer": "rebuilt_english",
                "text": text
            }
        )
        return [token["token"] for token in res.get("tokens", [])]

    def postings(self, term):
        res = helpers.scan(
            client=self._es,
            query={"query": {"match": {self._field: term}}},
            _source=False
        )
        return set(doc["_id"] for doc in res)

    def df(self, term):
        return self._es.count(
            index=self._index,
            body={"query": {"match": {self._field: term}}}
        ).get("count", 0)

    def tf(self, term, doc_id):
        doc = self._docs[doc_id]
        return 0 if doc["terms"].get(term) is None else doc["terms"][term]["term_freq"]

    def _term_vectors(self, doc_id):
        return self._es.termvectors(
            index=self._index,
            id=doc_id,
            fields=[self._field],
            field_statistics=False,
            term_statistics=False,
            positions=False,
            offsets=False
        )["term_vectors"][self._field]["terms"]

    def ttf(self, term):
        hits = self._es.search(
            index=self._index,
            body={"query": {"term": {"text": term}}},
            _source=False, size=1
        )["hits"]["hits"]
        if len(hits) == 0:
            return 1
        else:
            doc_id = hits[0]["_id"]
            return self._es.termvectors(
                index=self._index,
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

    def doc_count(self):
        return self._es.count(index=self._index)["count"]

    def unique_word_count(self):
        return self._es.search(
            index=self._index,
            body={"aggs": {"unique_word_count": {"cardinality": {"field": self._field}}}, "size": 0}
        )["aggregations"]["unique_word_count"]["value"]

    def sum_ttf(self):
        return self._es.termvectors(
            index=self._index,
            id="AP890101-0001",
            fields=[self._field]
        )["term_vectors"][self._field]["field_statistics"]["sum_ttf"]

    def doc_length(self, doc_id):
        return self._docs[doc_id]["doc_len"]

    def cache_docs(self, ids):
        def build_doc_struct(_doc):
            return {
                "doc_id": _doc["_id"],
                "terms": _doc["term_vectors"]["text"]["terms"],
                "doc_len": sum(stats["term_freq"]
                               for stats in
                               _doc["term_vectors"]["text"]["terms"].values())
            }

        ids = list(set(ids) - set(self._docs.keys()))
        print(f"    fetch_docs={len(ids)}(documents)")
        fetching_docs_start_time = time.time()
        size = 50
        p = 0
        while p < len(ids):
            res = self._es.mtermvectors(
                index=self._index,
                fields=["text"],
                ids=ids[p:p + size],
                term_statistics=False,
                field_statistics=False,
                offsets=False,
                positions=False
            )
            for doc in res["docs"]:
                self._docs[doc["_id"]] = build_doc_struct(doc)
            p += size
        print(f"    elapsed_t={time_used(fetching_docs_start_time)}")
        print(f"    ...cached {len(ids)} docs...")

    def search(self, **kwargs):
        return self._es.search(index=self._index, **kwargs)


if __name__ == '__main__':
    es = ESearch(
        hostname="http://localhost:9200",
        index="ap_dataset",
        field="text"
    )

    query = "a prediction about the prime lending rate, or will report an actual prime rate move"
    print(es.tokenize(text=query))
    esset = es.postings("china")
    print(es.df("china"))
    print(es.tf("china", "AP890707-0002"))
    print(es.ttf("china"))
    print(es.doc_count())
    print(es.sum_ttf())
    print(es.doc_length("AP890707-0002"))
