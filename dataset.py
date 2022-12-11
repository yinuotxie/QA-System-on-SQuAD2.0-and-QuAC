"""
Load dataset from the json file
"""
import urllib.request
import json 


def load_dataset(url):
    """
    Load the dataset from the url. Return context, queries, and answers.
    """
    with urllib.request.urlopen(url) as url:
        data_dict = json.load(url)

    texts = []
    queries = []
    answers = []

    # Search for each passage, its question and its answer
    for group in data_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    # get the answer end idx 
                    answer['answer_end'] = answer['answer_start'] + len(answer['text'])

                    # Store every passage, query and its answer to the lists
                    texts.append(context)
                    queries.append(question)
                    answers.append(answer)

    return texts, queries, answers