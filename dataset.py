"""
Load dataset from the json file
"""
import json
import urllib.request


def load_dataset(url):
    """
    Load the dataset from the url.
    """
    with urllib.request.urlopen(url) as url:
        data_dict = json.load(url)

    texts = []
    queries = []
    answers = []

    # Search for each passage, its question and its answer
    for group in data_dict['data']:
        for passage in group['paragraphs']:
            for qa in passage['qas']:
                texts.append(passage['context'])
                queries.append(qa['question'])
                results = []
                for answer in qa['answers']:
                    # get the answer end idx
                    answer['answer_end'] = answer['answer_start'] + len(answer['text'])
                    results.append(answer)

                answers.append(results)

    return texts, queries, answers
