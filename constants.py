BASE_SETTING = {
    "number_of_shards": 1,
    "number_of_replicas": 1,
    "analysis": {
        "filter": {
            "english_stop": {
                "type": "stop",
                "stopwords_path": "stoplist.txt"
            },
            "english_stemmer": {
                "type": "stemmer",
                "language": "english"
            },
        },
        "analyzer": {
            "rebuilt_english": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [
                    "lowercase",
                    "english_stop",
                    "english_stemmer"
                ]
            },
        }
    }
}

AP_DATASET_MAPPING = {
    "properties": {
        "text": {
            "type": "text",
            "fielddata": True,
            "analyzer": "rebuilt_english",
            "index_options": "positions"
        }
    }
}

AP_DATASET_INDEX_CONFIG = {
    "settings": BASE_SETTING,
    "mappings": AP_DATASET_MAPPING
}
