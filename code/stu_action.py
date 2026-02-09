import re
from config import SIM_PARAMS


class AgentAction:
    def __init__(self, profile, memory,llm):
        self.profile = profile
        self.memory = memory
        self.llm = llm


    def simulate_step(self, practice, time_step, teacher_predict_answer, similarity_fn):
        short_mem = self.memory.retrieve_short()
        long_mem = self.memory.retrieve_long()
        prompt = self._build_prompt(practice, short_mem, long_mem)
        messages = [
            {'role': 'system', 'content': self.profile.build_prompt()},
            {'role': 'user', 'content': prompt}
        ]
        resp = self.llm(messages)

        ans = {}
        for i in range(1, 5):
            m = re.search(rf'Task\s*{i}\s*:\s*(.+?)(?:\n|$)', resp, flags=re.I)
            if m:
                ans[f'task{i}'] = m.group(1).strip()

        print('ans:', ans)
        
        score = 0 if ans['task4'] == 'No' else 1

        record = [
            practice['exer_content'],
            practice['know_name'],
            score,
            1
        ]
        self.memory.write_factual(record)

        if SIM_PARAMS['learning_effect'] == 'yes':
            self.memory.reinforce(record)

        if SIM_PARAMS['forgetting_effect'] == 'yes':
            self.memory.forget(time_step)

        self.memory.retrieve_short()

        corrective = self.memory.reflect_corrective(practice, teacher_predict_answer, ans, score)

        reflect = self.memory.reflect_summary(corrective)
        
        messages.append({"role": "assistant",
                         "content": resp})
        messages.append({"role": "user",
                         "content": reflect})
        summary = self.llm(messages)
        
        self.memory.write_long_summary(summary)

        return ans, resp, corrective, summary

    def _build_prompt(self, practice, short_mem, long_mem):
        prompt = (
            f"Recommended Exercise:\n"
            f"- Textual Content: {practice['exer_content']}\n"
            f"- Knowledge Concept (true): {practice['know_name']}\n"
        )

        prompt += (
            "\nTask 1: Based on your Profile and Knowledge Proficiency, "
            "decide whether you want to attempt this exercise. "
            "If too difficult, output 'No'; otherwise, output 'Yes'.\n"
        )

        if short_mem:
            prompt += "\nYour Short-term Memory (recent facts):\n"
            for idx, r in enumerate(short_mem, 1):
                prompt += (
                    f" Record {idx}: Content='{r[0]}', Concept={r[1]}, Correct={r[2]}\n"
                )
            prompt += "\n"

        prompt += (
            "Task 2: Identify the knowledge concept tested by this exercise. "
            "Choose one from the following options:\n"
        )
        options = [practice['know_name']] + long_mem.get('practiced_knowledge', [])[:2]
        for opt in options:
            prompt += f" - {opt}\n"
        prompt += "Only output the concept name.\n"

        if long_mem.get('significant_facts'):
            prompt += "\nYour Long-term Memory (reinforced facts):\n"
            for idx, f in enumerate(long_mem['significant_facts'], 1):
                prompt += f" Record {idx}: Concept={f[1]}, ReinforcedTimes={f[3]}\n"
        if long_mem.get('learning_status'):
            prompt += (
                f"\nYour current Learning Status Summary: "
                f"{long_mem['learning_status'][-1]}\n"
            )

        prompt += (
            "\nTask 3: Propose a concise problem-solving idea based on your Profile and Memories, "
            "then give a final answer.\n"
        )

        prompt += (
            "\nTask 4: Predict whether you will answer correctly ('Yes' or 'No') based on the idea.\n"
        )

        prompt += (
            "\nFour tasks must output, Output format exactly as:\n"
            "Task1: <Yes/No>\n"
            "Task2: <concept>\n"
            "Task3: <your idea and final answer>\n"
            "Task4: <Yes/No>\n"
        )
        return prompt
