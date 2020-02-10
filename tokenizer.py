import re
import time
import parser

BASIC_TOKEN_REGEX = re.compile(r"[\w]*[a-zA-Z0-9_.]*[\w]")
LAST_TOKEN_ID = 0


def time_used(start):
    return f"{round(time.time() - start, 2)}s"


def build_tokenizer(regex=BASIC_TOKEN_REGEX, exclude=None, stemmer=None, lower_case=True):
    if exclude is None:
        exclude = set()

    def _tokenize(text):
        tokens = []
        for match in re.findall(regex, text):
            if lower_case:
                match = match.lower()
            if match in exclude:
                continue
            if stemmer is not None:
                match = stemmer.stem(match)
            tokens.append(match)
        return tokens

    return _tokenize


def create_token_tuples(tokens, start):
    return [(token, start + pos, pos) for pos, token in enumerate(tokens, start=1)]


def tokenize_in_batch(dir_path, fn_regex, limit, doc_id_map, token_id_map, doc_len_map, exclude=None, stemmer=None):
    global LAST_TOKEN_ID

    tokens_in_batch = list()
    limit_counter = 0
    docs_gen = parser.parse_files(dir_path, fn_regex, parser.ap_dataset_file_parser)
    tokenizer = build_tokenizer(regex=BASIC_TOKEN_REGEX, stemmer=stemmer, exclude=exclude)
    while True:
        try:
            doc = next(docs_gen)
        except StopIteration:
            if 0 < limit_counter < limit:
                yield tokens_in_batch
            break
        else:
            doc_id = doc_id_map[doc["_id"]]
            tokens = tokenizer(doc["text"])

            doc_len_map[doc_id] = len(tokens)

            for pos, token in enumerate(tokens, start=1):
                if token not in token_id_map:
                    LAST_TOKEN_ID += 1
                    token_id_map[token] = LAST_TOKEN_ID
                tokens_in_batch.append((token_id_map[token], doc_id, pos))
            limit_counter += 1

            if limit_counter == limit:
                yield tokens_in_batch
                limit_counter = 0
                tokens_in_batch = list()


def main():
    pass


if __name__ == '__main__':
    main()
