import re
import os
import constants
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


# parse documents
def file_parser(file_path, metadata=None):
    if metadata is None:
        metadata = dict()
    with open(file_path, "r") as fp:
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


def create_index(index_name, es_instance, index_config):
    try:
        if not es_instance.indices.exists(index_name):
            es_instance.indices.create(index=index_name, body=index_config)
            print('Created Index')
    except Exception as ex:
        raise ex


def batch_files(dir_path, metadata=None):
    if metadata is None:
        metadata = dict()
    files = sorted(os.listdir(dir_path))
    for file in files:
        abs_path = os.path.join(dir_path, file)
        if os.path.isfile(abs_path):
            yield from file_parser(abs_path, metadata)


def main(hostname, data_folder, index_name, index_config):
    metadata = {"_index": index_name}
    # connect to the ElasticSearch cluster, running on the default port 9200
    client = Elasticsearch(hostname)
    create_index(index_name=index_name, es_instance=client, index_config=index_config)
    # parse files and send parsed files to the ElasticSearch instance
    helpers.bulk(client=client, actions=batch_files(data_folder, metadata))
    print("Docs are indexed in ElasticSearch")


if __name__ == '__main__':
    host = "http://localhost:9200"
    docs_location = "/Users/tianzerun/Desktop/hw1/data/AP_DATA/ap89_collection"
    index = "ap_dataset"
    config = constants.AP_DATASET_INDEX_CONFIG
    main(host, docs_location, index, config)
