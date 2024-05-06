import time
from typing import List, Literal, Optional, Tuple, TypedDict, Dict
import re
import torch
from tqdm import tqdm

Role = Literal["system", "user", "assistant"]
from transformers import (
                          AutoModel,
                          AutoTokenizer,
                          AutoModelForCausalLM,)


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class CodeGen:
    @staticmethod
    def build(
            ckpt_dir: str,
            num_student_model: int,
            seed: int = 1,
    ) -> "CodeGen":
        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()
        checkpoint = ckpt_dir
        models = []
        # tokenizer = None
        tokenizer = AutoTokenizer.from_pretrained("{}/s_model_0".format(checkpoint))
        tokenizer.pad_token_id = tokenizer.eos_token_id
        for idx in range(num_student_model):
            model = AutoModelForCausalLM.from_pretrained("{}/s_model_{}".format(checkpoint, idx)).cuda()
            models.append(model)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return CodeGen(models, tokenizer)

    def __init__(self, models: List[AutoModel], tokenizer: AutoTokenizer):
        self.models = models
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens: List[List[int]],
            max_gen_len: int,
            temperature: float = 0.2,
            top_p: float = 0.95,
            logprobs: bool = False,
            echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

        params = self.models[0].config
        prompt_len = len(prompt_tokens)
        assert prompt_len <= params.max_position_embeddings
        total_len = min(params.max_position_embeddings, max_gen_len + prompt_len)

        pad_id = 0
        tokens = torch.full((1, total_len), pad_id, dtype=torch.long, device="cuda")

        for k, t in enumerate(prompt_tokens):
            tokens[:, k] = t
        eos_reached = torch.tensor([False] * 1, device="cuda")
        input_text_mask = tokens != pad_id

        for cur_pos in range(prompt_len, total_len):
            logits_all = []
            with torch.no_grad():
                for model in self.models:
                    logits = model(input_ids=tokens[:, 1:cur_pos]).logits
                    logits_all.append(logits)
                logits_sum = logits_all[0]
                for idx in range(1, len(logits_all)):
                    logits_sum += logits_all[idx] * 0.5
            if temperature > 0:
                probs = torch.softmax(logits_sum[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    next_token == self.tokenizer.eos_token_id
            )
            if all(eos_reached):
                break

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens)
            toks = toks[start: len(prompt_tokens) + max_gen_len]
            probs = None

            # cut to eos tok if any
            if self.tokenizer.eos_token_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_token_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
            self,
            prompts: List,
            temperature: float = 0.2,
            top_p: float = 0.95,
            max_gen_len: Optional[int] = None,
            logprobs: bool = False,
            echo: bool = False,
    ):
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        generated_texts = []
        for prompt in tqdm(prompts):
            prompt_tokens = self.tokenizer.encode(prompt)

            generation_tokens, generation_logprobs = self.generate(
                prompt_tokens=prompt_tokens,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                echo=echo,
            )

            EOF_STRINGS = ["def", "<\|endoftext\|>"]
            generated_text = prompt + self.tokenizer.decode(generation_tokens[0])
            string = re.split("(%s)" % "|".join(EOF_STRINGS), generated_text)
            generated_text = "".join(string[:3])

            data = {
                "text": generated_text,
            }
            generated_texts.append(data)

        return generated_texts


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
