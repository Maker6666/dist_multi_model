import fire
import json
from codegen.generation import CodeGen


def main(
        ckpt_dir: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_student_model: int = 1,
        max_gen_len: int = 256,
        num_samples: int = 1,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.2.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        num_student_model: The number of models it uses for inference, (Optional 1, 2, 3)
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 256.
        num_samples: Number of samples generated per prompt.
    """
    generator = CodeGen.build(
        ckpt_dir=ckpt_dir,
        num_student_model=num_student_model,
    )
    # Use huggingface datasets
    import datasets

    # Load evaluation dataset and metric
    human_eval = datasets.load_from_disk('./saved_humaneval/')

    # Generate completions for evaluation set
    n_tasks = len(human_eval["test"])

    prompts = []
    for task in range(n_tasks):
        prompt = human_eval["test"][task]["prompt"].strip()
        prompts.extend([prompt] * num_samples)

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    with open("inference_result", "w") as f_out:
        for item in results:
            f_out.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
