import sys
import time
import json
import zlib
from os import linesep

from nltk.stem.snowball import EnglishStemmer

import match
import search
import parser
from indexer import Catalog

# Available models to use for the purpose of scoring documents given a query.
# Do not change the representations arbitrarily as they map to dir names.
M_BM_25 = "BM-25"
M_ES_BUILT_IN = "ES-BUILT-IN"
M_TF = "TF"
M_TF_IDF = "TF-IDF"
M_ULM_LAPLACE = "ULM-LAPLACE"
M_ULM_JM = "ULM-JM"
M_PROXIMITY_SEARCH = "PROXIMITY-SEARCH"

MODEL_MAPPING = {
    M_ES_BUILT_IN: match.es_built_in_model,
    M_TF: match.tf_model,
    M_TF_IDF: match.tf_idf_model,
    M_BM_25: match.bm_25_model,
    M_ULM_LAPLACE: match.laplace_smoothing_language_model,
    M_ULM_JM: match.jelinek_mercer_smoothing_language_model,
    M_PROXIMITY_SEARCH: match.proximity_search_model
}


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


def create_zsearch_client(index_fp, index_folder, use_stemmer=True, use_compressor=False):
    stop_list_path = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/stoplist.txt"
    doc_list_path = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/doclist.txt"

    catalog = Catalog.load(f"{index_folder}/catalog")

    with open(f"{index_folder}/token_id_map.json", "r") as fp:
        token_id_map = json.load(fp)

    with open(f"{index_folder}/doc_len_map.json", "r") as fp:
        doc_len_map = json.load(fp)

    args = {
        "index_fp": index_fp,
        "catalog": catalog,
        "term_ids_map": token_id_map,
        "doc_ids_map": parser.load_doc_list(doc_list_path),
        "doc_len_map": doc_len_map,
        "exclude": parser.load_stop_words(stop_list_path)
    }

    if use_stemmer:
        args["stemmer"] = EnglishStemmer()

    if use_compressor:
        args["decompressor"] = zlib.decompress

    return search.ZSearch(**args)


def _run_models(client, query_level, *models):
    query_file = f"/Users/tianzerun/Desktop/CS6200/hw1/data/AP_DATA/query/{query_level}.txt"
    eval_folder = "./evaluation"
    result_filename = f"{client.__class__.__name__}_ranking.txt"

    match.get_metadata(client)
    tokenized_queries = {key: client.tokenize(value)
                         for key, value in match.read_queries(query_file).items()}

    for model in models:
        result_file_path = f"{eval_folder}/{model}/{query_level}/{result_filename}"
        # truncating the result file if it exists
        with open(result_file_path, "w") as out:
            match.run_engine_with(
                model=MODEL_MAPPING[model],
                client=client,
                queries=tokenized_queries,
                out_fp=out
            )


def _main():
    engine_start_time = time.time()
    models = (M_TF, M_TF_IDF, M_BM_25, M_ULM_LAPLACE, M_ULM_JM, M_PROXIMITY_SEARCH)

    # Choose client name and query level to
    client_name = search.ZSearch.__class__.__name__
    query_level = "l2"

    index_fp = None
    if client_name == search.ZSearch.__class__.__name__:
        index_folder = "./index"
        index_fp = open(f"{index_folder}/1-85", "rb")
        args = {
            "index_fp": index_fp,
            "index_folder": index_folder,
            "use_stemmer": True,
            "use_compressor": False
        }
        client = create_zsearch_client(**args)
    elif client_name == search.ESearch.__class__.__name__:
        args = {
            "hostname": "http://localhost:9200",
            "index": "ap_dataset",
            "field": "text"
        }
        client = search.ESearch(**args)
    else:
        raise ValueError

    _run_models(client, query_level, *models)

    if index_fp is not None:
        index_fp.close()

    print(f"Total time: {time_used(engine_start_time)}{linesep}")
    sys.exit()


if __name__ == '__main__':
    _main()
