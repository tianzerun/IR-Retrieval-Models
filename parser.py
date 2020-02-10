import re
import os
import constants
from bidict import bidict
from elasticsearch import Elasticsearch, helpers
from collections import namedtuple

Tag = namedtuple("Tag", "open close")


def create_tag(name):
    return Tag(f"<{name}>", f"</{name}>")


DOC = create_tag("DOC")
DOCNO = create_tag("DOCNO")
TEXT = create_tag("TEXT")
DOCNO_REGEX = re.compile(rf"^{DOCNO.open}(.+){DOCNO.close}$")
TEXT_OPEN_RE = re.compile(rf"\w*{TEXT.open}(.*)")


def ap_dataset_file_parser(fp, metadata=None):
    if metadata is None:
        metadata = dict()
    cur_doc_id = None
    content = []
    text_begin = False
    try:
        for line in fp:
            line = line.rstrip()
            if line == DOC.open:
                cur_doc_id = None
                content = []
            elif line == DOC.close:
                doc = {
                    "_id": cur_doc_id,
                    "text": " ".join(content)
                }
                doc.update(metadata)
                yield doc
            elif re.search(TEXT_OPEN_RE, line):
                text_begin = True
                text = re.search(TEXT_OPEN_RE, line).group(1)
                content.append(text)
            elif line == TEXT.close:
                text_begin = False
            elif text_begin:
                content.append(line)
            else:
                match = re.search(DOCNO_REGEX, line)
                if match:
                    doc_id = match.group(1).strip()
                    cur_doc_id = doc_id
    except UnicodeDecodeError as error:
        raise error


def parse_file(file_path, parser, metadata=None):
    with open(file_path, "r") as fp:
        yield from parser(fp, metadata)


def parse_files(dir_path, filename_regex, parser, metadata=None):
    files = list(filter(lambda name: re.match(filename_regex, name), sorted(os.listdir(dir_path))))
    for file in files:
        abs_path = os.path.join(dir_path, file)
        if os.path.isfile(abs_path):
            yield from parse_file(abs_path, parser, metadata)


def load_stop_words(path):
    stop_words = set()
    with open(path, "r") as fp:
        for line in fp:
            stop_words.add(line.strip().lower())
    return stop_words


def load_doc_list(path):
    doc_id_pair_regex = re.compile(r"(\d+)\s+(AP\d{6}-\w{4})")
    doc_list = dict()
    with open(path, "r") as fp:
        for line in fp:
            match = re.search(doc_id_pair_regex, line)
            if match is not None:
                doc_list[match.group(2)] = int(match.group(1))
    return bidict(doc_list)


def create_index(index_name, es_instance, index_config):
    try:
        if not es_instance.indices.exists(index_name):
            es_instance.indices.create(index=index_name, body=index_config)
            print('Created Index')
    except Exception as ex:
        raise ex


def _main():
    host = "http://localhost:9200"
    docs_location = "/Users/tianzerun/Desktop/CS6200/hw1/data/AP_DATA/ap89_collection"
    index = "ap_dataset"
    config = constants.AP_DATASET_INDEX_CONFIG
    ap_file_regex = re.compile(r"ap89\d{4}")

    metadata = {"_index": index}
    # connect to the ElasticSearch cluster, running on the default port 9200
    client = Elasticsearch(host)
    create_index(index_name=index, es_instance=client, index_config=config)
    # parse files and send parsed files to the ElasticSearch instance
    helpers.bulk(client=client, actions=parse_files(docs_location, ap_file_regex, ap_dataset_file_parser, metadata))
    print("Docs are indexed in ElasticSearch")


if __name__ == '__main__':
    _main()
