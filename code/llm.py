import os
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
import logging

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_llm(model_dir=".../Meta-Llama-3-8B-Instruct"):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)

    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        local_files_only=True,
        torch_dtype=torch.float16
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_dir,
        legacy=False,
        local_files_only=True
    )

    return model, tokenizer


class LlamaLLM:
    def __init__(self, model_dir=".../Meta-Llama-3-8B-Instruct", *args, **kwargs):

        self.model_dir = model_dir
        self.model = LlamaForCausalLM.from_pretrained(
            model_dir,
            device_map='auto',
            local_files_only=True,
            torch_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            local_files_only=True
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def __call__(self, prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            outputs = self.pipeline(
                messages,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=1.0,
                top_p=0.5,
            )
            return outputs[0]["generated_text"][-1]['content']
        except Exception as e:
            return self._simple_generate(prompt)

    def _simple_generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_responses(self, prompts):
        return [self(prompt) for prompt in prompts]


if __name__ == '__main__':
    llm = LlamaLLM()
    test_prompt = "Hello, how are you?"
    response = llm(test_prompt)
    print("test:", response[:100])