# Question Answering on SQuAD2.0 and SQuAD. 
## Introduction
Please see the report for the details of the project.
## Usage
Download the required package.
```
pip install transformers
pip install -U sentence-transformers rank_bm25
```
Clone the code from the github. 
```
git clone https://github.com/yinuotxie/QA-System-on-SQuAD2.0-and-QuAC.git
cd QA-System-on-SQuAD2.0-and-QuAC
```
Run the main file to see examples.
```
python3 main.py 
```
Run the main file to generate evuations.
```
python3 main.py --eval
```
Other userful arguments.
```
ptional arguments:
  -h, --help            show this help message and exit
  --dataset {squad,quac}
                        which dataset to use to eval
  --model_name MODEL_NAME
                        model name
  --retriever {lexical,semantic}
                        methods of retriever
  --apply_retriever     whether to use retriever
  --baseline            run the simple baseline
  --print_examples      Print examples
  --eval                whether to eval the pipeline.
```

