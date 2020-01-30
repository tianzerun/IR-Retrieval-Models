## CS6200 - HW1 - Retrieval Models

### High-level Requirements
Implement and compare various retrieval systems using vector space models and language models. 
There are two major components in this project:
- A program ([parser.py](./parser.py)) to parse the corpus and index it with ElasticSearch
- A query processor ([match.py](./match.py)), which runs queries from an input file using a selected retrieval model


### Challenges
- Familiarize with index creation and use ES APIs efficiently
- Implement scoring functions and tune parameters suitable for the ap dataset
- Manually modify the queries as the original queries are in natural language
- Make sense of the differences in performance (avg. precision) of VSMs and LMs
- Implement automatic query expansion (EC1 and EC2)


### Query Modification
In the `/query` folder, there are five files labeled l0.txt, l1.txt ... l4.txt. 
- l0: the unmodified queries
- l1: the most basic modification where I deleted obviously unrelated terms
- l2: aggressive manual modification where I deleted and added terms based on content of documents ranked on the top after running with l1.
- l3: pseudo-relevance feedback where I followed a very simple heuristic to first find relevant terms:
    ```
    # A term is important to a document if the term frequency is greater than the average term frequency. 
    # In other words, we think a term is important to a document when the document uses the term more
    # often than the term is used on average in other documents which also used the term.
    important_words = [term for term in terms if tf(term) > (ttf(term) / df(term))]
    ```
- l4: pseudo-relevance feedback using ES aggs "significant terms"

On average, the retrieval precision increases in the order: l0 < l1 < l4 < l2 < l3.

### MISC
- Assignment [Link](http://www.ccs.neu.edu/home/vip/teach/IRcourse/1_retrieval_models/HW1/HW1.html)
