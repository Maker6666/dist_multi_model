import fire
from codegen.generation import CodeGen


def main(
        ckpt_dir: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_student_model: int = 3,
        max_gen_len: int = 256,
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
    """
    generator = CodeGen.build(
        ckpt_dir=ckpt_dir,
        num_student_model=num_student_model,
    )

    prompts = ["def hello_world():"]

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for res in results:
        print(res['text'])


if __name__ == "__main__":
    fire.Fire(main)
