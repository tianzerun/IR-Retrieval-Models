import time
import indexer
import tokenizer
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


class ZSearch(Search):
    def __init__(self, index_fp, catalog, term_ids_map, doc_ids_map, doc_len_map,
                 converter=indexer.TextProcessor, exclude=None, stemmer=None):
        self._index_fp = index_fp
        self._catalog = catalog
        self._term_ids_map = term_ids_map
        self._doc_ids_map = doc_ids_map
        self._doc_len_map = doc_len_map
        self._sum_ttf = sum(ttf for ttf in self._doc_len_map.values())
        self._converter = converter
        self._tokenizer = tokenizer.build_tokenizer(exclude=exclude, stemmer=stemmer)
        self._cached_inverted_list = dict()

    def _load_inverted_list(self, term):
        if term not in self._cached_inverted_list:
            term_id = self._term_ids_map.get(term, None)
            if term_id is not None:
                offset = self._catalog.get_offset(term_id)
                size = self._catalog.get_size(term_id)
                raw = indexer.read(self._index_fp, offset, size, self._converter.decoder())
                self._cached_inverted_list[term] = indexer.InvertedList.deserialize(raw)
            else:
                self._cached_inverted_list[term] = indexer.InvertedList.dummy()
        return self._cached_inverted_list[term]

    def tokenize(self, text):
        return self._tokenizer(text)

    def postings(self, term):
        return set(self._doc_ids_map.inverse[_id]
                   for _id in self._load_inverted_list(term).doc_ids)

    def df(self, term):
        return self._load_inverted_list(term).df()

    def tf(self, term, doc):
        persisted_doc_id = self._doc_ids_map[doc]
        return self._load_inverted_list(term).tf(persisted_doc_id)

    def ttf(self, term):
        return self._load_inverted_list(term).ttf()

    def doc_count(self):
        return len(self._doc_ids_map)

    def unique_word_count(self):
        return len(self._term_ids_map)

    def sum_ttf(self):
        return self._sum_ttf

    def doc_length(self, doc_id):
        doc_integer_id = self._doc_ids_map[doc_id]
        return self._doc_len_map[str(doc_integer_id)]

    def cache_docs(self, ids):
        pass

