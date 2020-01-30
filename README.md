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
    After finding important terms in top documents, sort them based on their presence in top documents (i.e. A term is ranked the highest if it appears in all top documents.)
- l4: pseudo-relevance feedback using ES aggs "significant terms"

On average, the retrieval precision increases in the order: l0 < l1 < l4 < l2 < l3.


### Metrics
|       |ES-BUILT-IN |TF         |TF-IDF     |BM-25      |ULM-LAPLACE |ULM-JM |
|:----: |:----------:|:---------:|:---------:|:---------:|:----------:|:-----:|
|**l1** |0.2438      |**0.1773** |**0.2548** |0.2431     |0.1778      |0.2020 |
|**l2** |0.3279      |0.2270     |**0.3337** |0.3195     |**0.2195**  |0.2897 |
|**l3** |0.3553      |0.3057     |0.3456     |**0.3495** |**0.2665**  |0.3335 |
|**l4** |0.2788      |0.2153     |0.2871     |**0.2903** |**0.2022**  |0.2621 |
*This table shows the uninterpolated average precision for all six models when running against queries modified at different levels explained above.*


### MISC
- Assignment [Link](http://www.ccs.neu.edu/home/vip/teach/IRcourse/1_retrieval_models/HW1/HW1.html)
