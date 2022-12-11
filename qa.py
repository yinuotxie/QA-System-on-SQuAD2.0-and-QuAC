"""
Document Reader File.
"""
import torch
import torch.nn as nn


class QuestionAnswering(nn.Module):
    def __init__(self, tokenizer, retriever, reader):
        super(QuestionAnswering, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.reader = reader.to(self.device)

    def train(self):
        pass

    def question_answer(self, question, display=True):
        self.reader.eval()

        with torch.no_grad():
            topk_context = self.retriever.search(question, top_k=5)

            if display:
                print("========================================\n")
                print("Search Results")
                for i, context in enumerate(topk_context):
                    print(f"Top {i}: {context}")
                print("========================================\n")

            inputs = self.tokenizer(question, topk_context[0], truncation='only_second',
                                    return_tensors="pt", max_length=512)
            outputs = self.reader(**inputs.to(self.device))
            best_prob = torch.max(outputs[0]).item() * torch.max(outputs[1]).item()
            answer_start = torch.argmax(outputs[0])
            answer_end = torch.argmax(outputs[1]) + 1
            input_ids = inputs["input_ids"].tolist()[0]

            for context in topk_context[1:]:
                inputs = self.tokenizer(question, context, truncation='only_second',
                                        return_tensors="pt", max_length=512)
                outputs = self.reader(**inputs.to(self.device))

                prob = torch.max(outputs[0]).item() * torch.max(outputs[1]).item()
                # update to the better answer
                if best_prob < prob:
                    best_prob = prob
                    answer_start = torch.argmax(outputs[0])
                    answer_end = torch.argmax(outputs[1]) + 1
                    input_ids = inputs["input_ids"].tolist()[0]

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start: answer_end]))

            return answer
