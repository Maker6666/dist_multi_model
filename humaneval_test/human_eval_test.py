import json
import multiprocessing
import os
import re
import argparse
from evaluate import load
from datasets import load_from_disk
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="HumanEval based on generated samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generated-file",
        type=str,
        help="Generated .jsonl file",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples generated per prompt",
    )
    return parser


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    num_workers = multiprocessing.cpu_count()

    # Load evaluation dataset and metric
    # human_eval = load_dataset("openai_humaneval")
    human_eval = load_from_disk('saved_humaneval/')
    code_eval_metric = load("code_eval")

    num_gen_per_task = args.num_samples
    generated_all = []

    with open(args.generated_file, 'r') as generated_file:
        for line in generated_file:
            try:
                json_data = json.loads(line)
            except Exception as e:
                print(line)
                print("An error occurred:", e)
            generated_all.append(json_data)

    # Generate completions for evaluation set
    n_tasks = len(human_eval["test"])
    generations, references = [], []
    for task in tqdm(range(n_tasks)):
        task_generations = []

        for sample in generated_all[task * num_gen_per_task:(task + 1) * num_gen_per_task]:
            task_generations.append(sample['text'])

        generations.append(task_generations)
        test_func = human_eval["test"][task]["test"]
        entry_point = f"check({human_eval['test'][task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)

    # Evaluate completions with "code_eval" metric
    pass_at_k, _ = code_eval_metric.compute(
        references=references, predictions=generations, num_workers=num_workers
    )
    print(f"Results: {pass_at_k}")


if __name__ == "__main__":
    main()
