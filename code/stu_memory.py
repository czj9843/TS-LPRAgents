import math, random
from config import SIM_PARAMS

class Memory:
    def __init__(self, KCG, know_course_list, know_name):
        self.factual = []
        self.short = []
        self.long = {
            'significant_facts': [],
            'learning_status': [],
            'knowledge_proficiency': [],
            'practiced_knowledge': []
        }
        self.threshold = SIM_PARAMS['long_term_thresh']
        self.short_size = SIM_PARAMS['short_term_size']
        self.forget_lambda = SIM_PARAMS['forget_lambda']
        self.KCG = KCG
        self.know_course_list = know_course_list
        self.know_name = know_name

    def write_factual(self, record):
        self.factual.append(record)

    def retrieve_short(self):
        self.short = self.factual[-self.short_size:]
        return self.short

    def retrieve_long(self):
        return {
            'significant_facts': self.long['significant_facts'],
            'learning_status':  self.long['learning_status'],
            'knowledge_proficiency': self.long['knowledge_proficiency'],
            'practiced_knowledge': self.long['practiced_knowledge']
        }

    def write_long_summary(self, summary):
        self.long['learning_status'].append(summary)
    
    def write_know(self, practice):
        if practice['know_name'].replace('\"','').strip().lower() not in self.long['practiced_knowledge']:
                self.long['practiced_knowledge'].append(practice['know_name'].replace('\"','').strip().lower())

        
    def summarize_similarity_llm_kcg(self, record):
        record1 = '# knowledge concept 1 #: ' + record[1] + '\n'
        sim = []
        for memory_element in self.factual:
            record2 = '# knowledge concept 2 #: ' + memory_element[1] + '\n'
            if (self.know_name[record[1].replace('\"','').strip().lower()], self.know_name[memory_element[1].replace('\"','').strip().lower()]) in self.KCG or (self.know_name[memory_element[1].replace('\"','').strip().lower()], self.know_name[record[1].replace('\"','').strip().lower()]) in self.KCG:
                sim.append(1)
            else:
                if self.know_course_list[record[1].replace('\"','').strip().lower()] == self.know_course_list[memory_element[1].replace('\"','').strip().lower()]:
                    random_float = random.uniform(0, 1)
                    if random_float > 0.8:
                        sim.append(1)
                else:
                    sim.append(0)
        return sim
        
    def summarize_similarity_llm(self, record):
        pass

    def reinforce(self, record):      
        sim_list = self.summarize_similarity_llm_kcg(record)
        for i, sim in enumerate(sim_list):
            self.factual[i][3] += sim
        max_count = max((r[3] for r in self.factual), default=1)
        self.factual.append([record[0], record[1], record[2], max_count])
        existing = {f[-1] for f in self.long['significant_facts']}
        for idx, rec in enumerate(self.factual):
            if rec[3]>=self.threshold and (idx+1) not in existing:
                rec.append(idx+1)
                self.long['significant_facts'].append(rec)
                rec[3]=1
        return sim_list

    def forget(self, t):
        kept=[]
        for fact in self.long['significant_facts']:
            ts=fact[4]
            if 1/(1+math.exp(-(t-ts)))<=self.forget_lambda:
                kept.append(fact)
            else:
                self.factual[ts][3]=1
        self.long['significant_facts']=kept

    def reflect_corrective(self, practice, teacher_predict_answer, ans, score):
        fb=''
        if practice['know_name']!=ans.get('task2'):
            fb += 'The knowledge tested by this question is ' + practice['know_name'].replace('\"','').strip().lower() + ' but you wrongly think the knowledge is ' + (ans.get('task2') or 'unknown knowledge').strip().lower() + '. \n'
              
        if (teacher_predict_answer!='1') and score==1:
            fb+="Teacher thought you couldn't solve this problem correctly, but in fact, you will solve it correctly. \n"
        if (teacher_predict_answer!='0') and score==0:
            fb += "Teacher thought you could solve this problem correctly, but in fact, you does not answer it correctly. \n"
        return fb

    def reflect_summary(self, corrective):
        if corrective != '':
            summary = corrective
        else:
            summary = ''
        summary+=f"\n You should directly output your reflection and summarize your # Learning Status # within 500 words based on your # profile #, # short-term memory #, # long-term memory # and previous # Learning Status #. Do not output any other information."

        self.long['learning_status'].append(summary)
        return summary