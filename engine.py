import sys
import time
import json
from os import linesep

from nltk.stem.snowball import EnglishStemmer

import match
import search
import parser
from indexer import Catalog

M_BM_25 = "BM-25"
M_ES_BUILT_IN = "ES-BUILT-IN"
M_TF = "TF"
M_TF_IDF = "TF-IDF"
M_ULM_LAPLACE = "ULM-LAPLACE"
M_ULM_JM = "ULM-JM"


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


def main(index_name, file_path, *models):
    stop_list_path = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/stoplist.txt"
    doc_list_path = "/Users/tianzerun/Desktop/hw2/data/AP_DATA/doclist.txt"
    eval_folder = "./evaluation/"
    hw = "hw2"

    model_mapping = {
        M_ES_BUILT_IN: match.es_built_in_model,
        M_TF: match.tf_model,
        M_TF_IDF: match.tf_idf_model,
        M_BM_25: match.bm_25_model,
        M_ULM_LAPLACE: match.laplace_smoothing_language_model,
        M_ULM_JM: match.jelinek_mercer_smoothing_language_model
    }

    catalog = Catalog.load("index/catalog")

    with open("index/token_id_map.json", "r") as fp:
        token_id_map = json.load(fp)

    with open("index/doc_len_map.json", "r") as fp:
        doc_len_map = json.load(fp)

    # client = ESearch(hostname="http://localhost:9200", index=index_name, field="text")

    index_fp = open("index/1-85", "rb")
    client = search.ZSearch(
        index_fp=index_fp,
        catalog=catalog,
        term_ids_map=token_id_map,
        doc_ids_map=parser.load_doc_list(doc_list_path),
        doc_len_map=doc_len_map,
        exclude=parser.load_stop_words(stop_list_path),
        stemmer=EnglishStemmer()
    )

    match.get_metadata(client)
    tokenized_queries = {key: client.tokenize(value)
                         for key, value in match.read_queries(file_path).items()}
    for model in models:
        match.run_engine_with(
            model=model_mapping[model],
            client=client,
            queries=tokenized_queries,
            result_file=f"{eval_folder}/{model}/{query_level}/ranking_{hw}.txt"
        )
    index_fp.close()


if __name__ == '__main__':
    query_level = "l2"
    query_file = f"/Users/tianzerun/Desktop/CS6200/hw1/data/AP_DATA/query/{query_level}.txt"
    engine_start_time = time.time()
    use_models = (M_TF, M_TF_IDF, M_BM_25, M_ULM_LAPLACE, M_ULM_JM)
    main("ap_dataset", query_file, M_TF_IDF)

    print(f"Total time: {time_used(engine_start_time)}{linesep}")
    sys.exit()
