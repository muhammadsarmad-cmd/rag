import json
from openai import OpenAI
from rag import retrieve_context,query
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

golden_dataset = [
    {
        "question": "How many days is recreation leave?",
        "expected_answer": "Recreation leave is 15 calendar days and in lieu thereof 10 days leave on full pay shall be debited to the leave account."
    },
    {
        "question": "What is the maximum leave on full pay without medical certificate?",
        "expected_answer": "The maximum period of leave on full pay without medical certificate is 120 days."
    },
    {
        "question": "How many days of maternity leave is a female civil servant entitled to?",
        "expected_answer": "A female civil servant is entitled to maternity leave on full pay for a maximum period of 90 days."
    },
    {
        "question": "What happens to leave balance when a civil servant quits service?",
        "expected_answer": "All leave at the credit of a civil servant shall lapse when he quits service."
    },
    {
        "question": "What is the maximum extraordinary leave that can be granted at one time?",
        "expected_answer": "Extraordinary leave can be granted up to a maximum of 5 years at a time, provided the civil servant has been in continuous service for not less than 10 years."
    },
    {
    "question": "Can extraordinary leave be combined with leave on full pay?",
    "expected_answer": "The maximum period of five years shall be reduced by the period of leave on full pay or half pay if granted in combination with extraordinary leave."
    },
    {
        "question": "What is the penalty for absence on unsanctioned leave?",
        "expected_answer": "Double the period of absence shall be debited against the leave account and one day salary shall be deducted."
    }
]

def eval_single(context:str,question:str,actual_answer:str,expected_answer:str):
    ai_response = client.chat.completions.create(
        model = "o4-mini-2025-04-16",
        messages = [
            {"role":"system","content":"You are a judge for evaluating rag system your task is to evaulate rag system and give faithfullnes,relevance and context recall score"},
            {"role" : "user","content" : 
             f"""Given:
Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}
Retrieved Context: {context}

Score the following metrics from 0 to 1:
- faithfulness: is the actual answer grounded in the retrieved context?
- relevance: does the actual answer address the question?
- context_recall: does the retrieved context contain enough info to answer the question?

Return ONLY a JSON object with keys: faithfulness, relevance, context_recall, reasoning
"""
             
             },
            
        ],
    )

    return json.loads(ai_response.choices[0].message.content)

def eval_pipeline():
    faithfulness = []
    relevance = []
    context_recall = []

    for item in golden_dataset:
        context = retrieve_context(item["question"])
        actual_answer = query(item["question"])
        eval_reponse = eval_single(context,item["question"],actual_answer,item["expected_answer"])
        faithfulness.append(eval_reponse['faithfulness'])
        relevance.append(eval_reponse['relevance'])
        context_recall.append(eval_reponse['context_recall'])
        print(f"Question : {item["question"]}")
        print(f"Score: {eval_reponse}")


    average_faithfulness = sum(faithfulness)/len(faithfulness)
    average_relevance = sum(relevance)/len(relevance)
    average_context_recall = sum(context_recall)/len(context_recall)

    print(f"Average Faithfulness: {average_faithfulness}")
    print(f"Average relevance: {average_relevance}")
    print(f"Average context_recall: {average_context_recall}")


    
eval_pipeline()
    
