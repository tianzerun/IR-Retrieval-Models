import re
import os
import zlib
import time
import json
import pickle
import parser
import shutil
import tokenizer
from io import SEEK_SET
from nltk.stem.snowball import EnglishStemmer


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


class Catalog(object):
    def __init__(self, index_path):
        self._index_path = index_path
        self._logs = dict()

    def __iter__(self):
        return iter(self._logs.keys())

    def __next__(self):
        return next(self)

    def __contains__(self, item):
        return item in self._logs

    @property
    def index_path(self):
        return self._index_path

    def add(self, key, offset, size):
        self._logs[key] = (offset, size)

    def get_offset(self, key):
        if key in self._logs:
            return self._logs[key][0]

    def get_size(self, key):
        if key in self._logs:
            return self._logs[key][1]

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as fp:
            return pickle.load(fp)


class TextProcessor(object):
    @classmethod
    def encoder(cls, compressor=None):
        def _encode(text):
            bin_text = text.encode(encoding='utf-8', errors='strict')
            if compressor is not None:
                return compressor(bin_text)
            return bin_text

        return _encode

    @classmethod
    def decoder(cls, decompressor=None):
        def _decode(raw):
            if decompressor is not None:
                raw = decompressor(raw)
            return raw.decode(encoding='utf-8', errors='strict')

        return _decode


class DeltaEncoding(object):
    @classmethod
    def encoder(cls):
        def _encode(nums):
            if len(nums) == 0:
                return nums
            else:
                ans = [nums[0]]
                for i in range(1, len(nums)):
                    ans.append(nums[i] - nums[i - 1])
            return ans

        return _encode

    @classmethod
    def decoder(cls):
        def _decode(nums):
            if len(nums) == 0:
                return nums
            else:
                ans = [nums[0]]
                for i in range(1, len(nums)):
                    ans.append(nums[i] + ans[-1])
            return ans

        return _decode


class Posting(object):
    """
    e.x.    1,3,1,2,3
            2,5,1,2,3,4,5
    """

    def __init__(self, doc_id: int, positions: [int]):
        self._doc_id = doc_id
        self._positions = positions
        self._tf = len(self._positions)

    @property
    def doc_id(self):
        return self._doc_id

    def tf(self):
        return self._tf

    def serialize(self, encoding_on=True):
        if encoding_on:
            positions = DeltaEncoding.encoder()(self._positions)
        else:
            positions = self._positions
        return f"{self._doc_id},{self.tf()},{','.join([str(pos) for pos in positions])}"

    @classmethod
    def deserialize(cls, text: str, decoding_on=True):
        [doc_id, _, *positions] = text.split(",")
        positions = [int(pos) for pos in positions]
        if decoding_on:
            positions = DeltaEncoding.decoder()(positions)
        return Posting(int(doc_id), positions)


class InvertedList(object):
    """
    e.x. 1,3,1,2,3|2,5,1,2,3,4,5
    """
    splitter = "="
    posting_splitter = "|"

    def __init__(self, term: int, postings):
        self._term = term
        self._postings = postings
        self._df = len(self._postings)
        self._ttf = sum(pos.tf() for pos in self._postings.values())

    @property
    def term(self):
        return self._term

    @property
    def postings(self):
        return self._postings

    @property
    def doc_ids(self):
        return set(_id for _id in self._postings.keys())

    def df(self):
        return self._df

    def tf(self, doc_id):
        if doc_id not in self._postings:
            return 0
        return self._postings[doc_id].tf()

    def ttf(self):
        return self._ttf

    def update(self, other):
        if self._term == other.term:
            # TODO notice there might be doc_id collisions in this update, handle explicitly as needed
            self._postings.update(other.postings)
        else:
            raise ValueError

    def serialize(self):
        return f"{self._term}" \
               f"{self.splitter}" \
               f"{self.posting_splitter.join([Posting.serialize(pos) for pos in self._postings.values()])}"

    @classmethod
    def deserialize(cls, text: str):
        [term, postings_text] = text.split(cls.splitter)
        postings_map = dict()
        for pos in postings_text.split(cls.posting_splitter):
            posting_obj = Posting.deserialize(pos)
            postings_map[posting_obj.doc_id] = posting_obj
        return InvertedList(int(term), postings_map)

    @classmethod
    def dummy(cls):
        return InvertedList(-1, dict())


class Indexer(object):
    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder
        self._catalogs = list()

    def index(self, batch_id, tokens):
        postings = dict()
        for term_id, doc_id, pos in tokens:
            if term_id in postings:
                if doc_id in postings[term_id]:
                    postings[term_id][doc_id].append(pos)
                else:
                    postings[term_id][doc_id] = [pos]
            else:
                postings[term_id] = {
                    doc_id: [pos]
                }

        inverted_index = dict()
        for term_id, postings in postings.items():
            inverted_index[term_id] = InvertedList(
                term=term_id,
                postings={doc_id: Posting(doc_id, positions) for doc_id, positions in postings.items()}
            )

        catalog = write(f"tmp/{batch_id}", inverted_index, self._encoder)
        self._catalogs.append(catalog)

    def merge_indexes(self):
        def index_filename_generator(f1, f2):
            f1name = f1.split("/")[-1]
            f2name = f2.split("/")[-1]
            if "-" in f1name:
                f1name = f1name.split("-")[0]
            if "-" in f2name:
                f2name = f2name.split("-")[-1]
            return f"tmp/{f1name}-{f2name}"

        while len(self._catalogs) > 1:
            merged = list()
            for i in range(0, len(self._catalogs) - 1, 2):
                c1 = self._catalogs[i]
                c2 = self._catalogs[i + 1]
                print(f"    indexes:{c1.index_path} - {c2.index_path}")
                merge_two_start_t = time.time()
                catalog = self._merge(c1, c2, index_filename_generator(c1.index_path, c2.index_path))
                print(f"    time_taken:{time_used(merge_two_start_t)}")
                merged.append(catalog)
            if len(self._catalogs) > 2 * len(merged):
                merged.append(self._catalogs[-1])
            self._catalogs = merged
        return self._catalogs.pop()

    def _merge(self, c1, c2, new_index_file_path):
        f1p = open(c1.index_path, "rb")
        f2p = open(c2.index_path, "rb")
        merged_catalog = Catalog(new_index_file_path)
        f3p = open(merged_catalog.index_path, "wb")

        offset = 0
        for term in c1:
            final_inverted_list = read(f1p, c1.get_offset(term), c1.get_size(term), self._decoder)

            if term in c2:
                tmp = InvertedList.deserialize(read(f2p, c2.get_offset(term), c2.get_size(term), self._decoder))
                combined = InvertedList.deserialize(final_inverted_list)
                combined.update(tmp)
                final_inverted_list = combined.serialize()

            size = write_term(f3p, offset, final_inverted_list, self._encoder)
            merged_catalog.add(term, offset, size)
            offset += size

        for term in c2:
            if term not in merged_catalog:
                content = read(f2p, c2.get_offset(term), c2.get_size(term), self._decoder)
                size = write_term(f3p, offset, content, self._encoder)
                merged_catalog.add(term, offset, size)
                offset += size

        f1p.close()
        f2p.close()
        f3p.close()
        return merged_catalog


def write_term(fp, offset, content: str, encoder, delimiter="\n"):
    fp.seek(offset, SEEK_SET)
    text = encoder(f"{content}{delimiter}")
    fp.write(text)
    return len(text)


def write(filename, index, encoder):
    offset = 0

    catalog = Catalog(filename)
    with open(filename, "wb") as fp:
        for term, postings in index.items():
            term_repr = postings.serialize()
            size = write_term(fp, offset, term_repr, encoder)
            catalog.add(term, offset, size)
            offset += size
    return catalog


def read(fp, offset, size, decoder, delimiter="\n"):
    fp.seek(offset, SEEK_SET)
    return decoder(fp.read(size)).strip(delimiter)


def create_dir(dir_path):
    try:
        os.mkdir(dir_path)
        print(f"Directory {dir_path} created ")
    except FileExistsError:
        print(f"Directory {dir_path} already exists")


def main():
    program_start_t = time.time()

    remove_tmp_folder = False
    compress = False

    tmp_folder = "./tmp/"
    index_folder = "./index/"
    create_dir(tmp_folder)
    create_dir(index_folder)

    ap_file_regex = re.compile(r"ap89\d{4}")
    ap_dataset_dir = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/ap89_collection"
    stop_list_path = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/stoplist.txt"
    doc_list_path = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/doclist.txt"

    token_id_map = dict()
    doc_len_map = dict()
    doc_ids = parser.load_doc_list(doc_list_path)
    exclude = parser.load_stop_words(stop_list_path)
    stemmer = EnglishStemmer()

    if compress:
        encoder = TextProcessor.encoder(zlib.compress)
        decoder = TextProcessor.decoder(zlib.decompress)
    else:
        encoder = TextProcessor.encoder()
        decoder = TextProcessor.decoder()

    indexer = Indexer(encoder, decoder)

    limit = 1000
    args = {
        "dir_path": ap_dataset_dir,
        "fn_regex": ap_file_regex,
        "limit": limit,
        "doc_id_map": doc_ids,
        "token_id_map": token_id_map,
        "doc_len_map": doc_len_map,
        "exclude": exclude,
        "stemmer": stemmer
    }

    tokenize_start_t = time.time()
    for batch_id, batch in enumerate(tokenizer.tokenize_in_batch(**args), start=1):
        print(f"Tokenize batch {batch_id}:")
        print(f"    files=[{batch[0][1]}...{batch[-1][1]}]")
        print(f"    elapsed_t={time_used(tokenize_start_t)}")

        print(f"Index batch {batch_id}:")
        indexing_start_t = time.time()
        indexer.index(batch_id, batch)
        print(f"    elapsed_t={time_used(indexing_start_t)}")

        tokenize_start_t = time.time()

    print("Merge indexes:")
    merge_indexes_start_t = time.time()
    catalog = indexer.merge_indexes()
    print(f"Total time taken for merging indexes: {time_used(merge_indexes_start_t)}{os.linesep}")

    print("Dump final catalog:")
    dump_catalog_start_t = time.time()
    with open("index/catalog", "wb") as fp:
        pickle.dump(catalog, fp)
    print(f"    elapsed_t={time_used(dump_catalog_start_t)}{os.linesep}")

    print("Dump token_id_map:")
    dump_token_id_start_t = time.time()
    with open("index/token_id_map.json", "w") as fp:
        json.dump(token_id_map, fp, separators=(',', ':'))
    print(f"    elapsed_t={time_used(dump_token_id_start_t)}{os.linesep}")

    print("Dump doc_length_map:")
    dump_doc_len_start_t = time.time()
    with open("index/doc_len_map.json", "w") as fp:
        json.dump(doc_len_map, fp, separators=(',', ':'))
    print(f"    elapsed_t={time_used(dump_doc_len_start_t)}{os.linesep}")

    print("Move final index from tmp to index folder:")
    shutil.move(catalog.index_path, f"index/{catalog.index_path.split('/')[-1]}")

    if remove_tmp_folder:
        print(f"Remove tmp folder...{os.linesep}")
        shutil.rmtree(tmp_folder)

    print(f"Total time taken: {time_used(program_start_t)}")


if __name__ == '__main__':
    main()
