## Download Student model
Our distilled student models are available. You can download from: https://huggingface.co/maker666/dist_multi_model/tree/main

## How to use
To perform model inference, test code generation capability, run (in senteval/data/downstream/):   
```javascript copy
python models_generate.py --ckpt_dir "../dist_multi_model"
```
You can customize the prompt content and  the number of students in models_generate.py
## Humaneval Evaluation
Evaluating the model using the humaneval dataset, run:
```javascript copy
python ./humaneval_test/humaneval_generate.py --ckpt_dir "../dist_multi_model" --num_samples 1
```
```javascript copy
python ./humaneval_test/humaneval_test.py -generated-file "inference_results" --num_samples 1
```
You can change the num_student_model parameter to test the performance of multiple student models.You can also change temperature top_p to select the best result
