import math
import re
import torch
import json
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import numpy as np
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Teacher:
    def __init__(self,llm):
        super().__init__()
        self.teacher_setting=f"You are an experienced math teacher with ten years of teaching experience."
        self.main_data_path = ".../dataset"
        self.llm = llm
        self.recommendation_history = []
        self.reflection_memory = []
        self.dataname = "assist"
        self.QTEXT = True
        self.REFLECTION = True

        if self.dataname == "assist":
            with open(self.main_data_path+f"/K_Directed.txt", "r") as file:
                lines = file.readlines()
            with open(self.main_data_path+f"/k2txt.json", 'r') as f:
                self.k2text = json.load(f)
            with open(self.main_data_path+f"/k2q.json", 'r') as f:
                self.k2q = json.load(f)
            with open(self.main_data_path+f"/q2k.json", 'r') as f:
                self.q2k = json.load(f)
            self.q2text = json.load(open(self.main_data_path+f"/q2txt.json", 'r'))
            
        else:
            print("only junyi / assist / TextLog are supported")
            raise NotImplementedError
        
        self.k_graph = [tuple(map(int, line.strip().split())) for line in lines]
        self.edutools = True 
        
        self.bert_path = r".../bert-master"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_model = BertModel.from_pretrained(self.bert_path)
        self.bert_model.to(device)
        self.bert_model.eval()
        self.compute_similarities()

    def compute_similarities(self):
    
        all_exercises = list(self.q2text.keys())
        exercise_embeddings = {}
    
        for ex_id in tqdm(all_exercises, desc="Exercises"):
            text = self.q2text[ex_id]
            embedding = self.get_bert_embedding(text)
            exercise_embeddings[ex_id] = embedding
    
        all_kcs = list(self.k2text.keys())
        kc_embeddings = {}
    
        for kc_id in tqdm(all_kcs, desc="KC"):
            text = self.k2text[kc_id]
            embedding = self.get_bert_embedding(text)
            kc_embeddings[kc_id] = embedding
    
        J = len(all_exercises)
        self.SE_dict = {}
    
        for ex_j in tqdm(all_exercises, desc="SE"):
            e_j = exercise_embeddings[ex_j]
            similarity_sum = 0
        
            for ex_i in all_exercises:
                e_i = exercise_embeddings[ex_i]
                similarity_sum += np.dot(e_j, e_i)
        
            self.SE_dict[ex_j] = similarity_sum / J
    
        self.SK_dict = {}
    
        for ex_j in tqdm(all_exercises, desc="SK"):
            if ex_j not in self.q2k:
                self.SK_dict[ex_j] = 0
                continue

            exercise_kcs = self.q2k[ex_j]
            N = len(exercise_kcs)
        
            if N == 0:
                self.SK_dict[ex_j] = 0
                continue
        
            total_similarity = 0
        
            for kc_m_id in exercise_kcs:
                if str(kc_m_id) not in kc_embeddings:
                    continue
                
                kc_m = kc_embeddings[str(kc_m_id)]
                kc_similarity_sum = 0
            
                for kc_i_id in exercise_kcs:
                    if str(kc_i_id) not in kc_embeddings:
                        continue
                    
                    kc_i = kc_embeddings[str(kc_i_id)]
                    kc_similarity_sum += np.dot(kc_m, kc_i)
            
                avg_similarity = kc_similarity_sum / len(exercise_kcs) if len(exercise_kcs) > 0 else 0
                total_similarity += avg_similarity
        
            self.SK_dict[ex_j] = total_similarity / N if N > 0 else 0

    
    def get_bert_embedding(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
        return embedding



    def find_one_hop_predecessors(self, node):
        one_hop_predecessors = set([edge[0] for edge in self.k_graph if edge[1] == node])
        return one_hop_predecessors

    def given_advise(self, step, history_text, ques_log, student_profile, goal_id, goal_state):
        with open('.../exer_difficulty.json', 'r', encoding='utf-8') as f:
            exer_diff_list = json.load(f)
        learning_goal = self.k2text[str(goal_id)]

        teacher_reflection_memory = self.reflection_memory

        if self.edutools == True:
            one_hop_predecessors = self.find_one_hop_predecessors(goal_id)

            if step == 0:
                last_exer_id = ques_log[-1]
                threshold = exer_diff_list[str(last_exer_id)] 
            else:
                threshold = teacher_reflection_memory[-1]['reflection']

            
            if self.QTEXT:
                candidate_questions_about_neighbor_knowledge = [{'exer_id':q, 'question_text':self.q2text[str(q)]} for k in one_hop_predecessors for q in self.k2q[str(k)] if q not in ques_log and exer_diff_list[str(q)] < threshold]
                candidates = [{'exer_id':q, 'question_text':self.q2text[str(q)]} for q in self.k2q[str(goal_id)] if q not in ques_log and exer_diff_list[str(q)] < threshold and exer_diff_list[str(q)] > 0.3]
                
            else:
                learning_goal = goal_id
                candidate_questions_about_neighbor_knowledge = [{'exer_id':q } for k in one_hop_predecessors for q in self.k2q[k] if q not in ques_log and exer_diff_list[str(q)] < threshold ]
                candidates = [{'exer_id':q} for q in self.k2q[str(goal_id)] if q not in ques_log and exer_diff_list[str(q)] < threshold and exer_diff_list[str(q)] > 0.3 ]

            if len(candidates) == 0:
                candidates = candidate_questions_about_neighbor_knowledge
                    
                if len(candidates) == 0:
                    if self.QTEXT:
                        candidates = [{'exer_id':q, 'question_text':self.q2text[str(q)]} for q in self.k2q[str(goal_id)] if exer_diff_list[str(q)] < 0.8 ] + [{'exer_id':q, 'question_text':self.q2text[str(q)]} for k in one_hop_predecessors for q in self.k2q[k] if q not in ques_log and exer_diff_list[str(q)] < threshold]
                    else:
                        candidates = [{'exer_id':q } for q in self.k2q[str(goal_id)] if exer_diff_list[str(q)] < 0.8 ] + [{'exer_id':q } for k in one_hop_predecessors for q in self.k2q[k] if q not in ques_log and exer_diff_list[str(q)] < threshold ]
                else:
                    return -1
        else:
            candidates_qid = random.sample(self.q2text.keys(), 10)
            candidates = [{'exer_id':q, 'question_text':self.q2text[str(q)]} for q in candidates_qid]
       
        if len(candidates) > 10:
            candidates = random.sample(candidates, 10) 

        candidate_scores = []
        for candidate in candidates:
            exer_id = str(candidate['exer_id'])
                 
            SE = self.SE_dict.get(exer_id, 0.5)
            SK = self.SK_dict.get(exer_id, 0.5)
            W_ij = math.sqrt(abs((threshold - exer_diff_list[str(exer_id)]) ** 2 - SE + SK))
            candidate_scores.append({
            'exer_id': candidate['exer_id'],
            'question_text': candidate.get('question_text', ''),
            'score': W_ij
        })
        
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        top_candidate = candidate_scores[:min(1, len(candidate_scores))]

        if self.REFLECTION:
            advise_prompt= f"""Given the following history text: {history_text[-5:]}, the student profile: {student_profile}, the knowledge learning goal: {learning_goal}. Here are the candidate question list: {candidates}, the top candidate question is {top_candidate}. 
Please provide a JSON object with the following structure:
{{
  "exer_id": <the id of the recommended question>,
  "recommend_reason": "<reason for recommending this question>",
  "predict_answer": <1 or 0>
}}
Only output the JSON object, nothing else."""
        else:
            advise_prompt=f"Given the following history text: {history_text[-5:]}, the student profile: {student_profile}, the knowledge learning goal: {learning_goal}. Here are the candidate question list: {candidates}, the top candidate question is {top_candidate}. Please provide the most suitable exer_id from above list, that can help the student to achieve the learning goal efficiently. For example, the output format should be :['exer_id': 'xxx'], except this format, please do not output anything."
  
        advise=self.llm(self.teacher_setting+advise_prompt)

        try:

            json_match = re.search(r'\{.*\}', advise, re.DOTALL)
            if json_match:
                advise_dict = json.loads(json_match.group())
                print('success:', advise_dict)
                return advise_dict
        except json.JSONDecodeError as e:
            print(f"JSON fail: {e}")

    
    def judge_answer(self, question, answer_content, ground_truth):
        judge_prompt=f"The question is [{question}]. Given the answer content: {answer_content}, the ground truth: {ground_truth}, please provide the judgement about the correctness of the answer. For example, the output format should be:['judge_result': 'True' or 'False']. "
        judge=self.llm(self.teacher_setting+judge_prompt)
        return judge

    def reflect_on_outcome(self, question_id, predict_answer, actual_outcome, student_feedback):
        record = {
            'question_id':question_id,
            'predicted': predict_answer,
            'actual': 'correct' if actual_outcome else 'wrong',
            'timestamp': len(self.recommendation_history)
        }
        self.recommendation_history.append(record)

        reflection = ""
        

        predicted_correct = predict_answer
        if predicted_correct == actual_outcome:
            reflection_diffcode = 0.5  
        elif predicted_correct and not actual_outcome:
            reflection_diffcode = 0.2 
        else:
            reflection_diffcode = 0.7 

        self.reflection_memory.append({
            'reflection': reflection_diffcode,
            'question_id': question_id,
            'step': len(self.recommendation_history)
        })

        if len(self.reflection_memory) > 5:
            self.reflection_memory.pop(0)
            

        return reflection
    
    def get_recommendation_summary(self):

        if not self.recommendation_history:
            return "None"
        
        correct_predictions = sum(1 for r in self.recommendation_history 
                                  if (r['predicted'].lower() in ['yes', 'true', 'correct']) == 
                                  (r['actual'] == 'correct'))
        
        total = len(self.recommendation_history)
        accuracy = correct_predictions / total if total > 0 else 0
        
        return accuracy