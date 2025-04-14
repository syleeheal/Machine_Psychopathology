# Machine_Psychopathology

This is the code repository of the paper ["Emergence of psychopathological computations in large language models"](https://arxiv.org/abs/2504.08016), containing the following:
1. [_code_]: input prompt designs for the synthetic data generation
2. [_code_]: S3AE training and evaluation
3. [_code_]: Q&A session design
4. [_code_]: causal inference
5. [_data_]: unit activation dataset
6. [_model_]: trained S3AE parameters
   
---

## S3AE
We provide codes related to **S**entence-level, **S**upervised, **S**parse **A**uto**E**ncoder (S3AE).

 - [s3ae.py]: S3AE architecture, load, and inference code 
 - [s3ae_main.py]: S3AE training and evaluation code

The trained S3AE weights are provided in this [HuggingFace Repo](https://huggingface.co/syleetolow/s3ae). The S3AE was trained on the residual stream in the 10th layer of instruction-tuned [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it), using a proprietary synthetic dataset with psychopathology symptom labels. The model weight precision is bfloat16, and the hidden dimension size is 8 times that of the LLM residual stream.

The 1st to 17th dimensions of S3AE hidden features, respectively, correspond to activations of the following thoughts:

    1: 'depressed mood', 
    2: 'anhedonia (loss of interest)',
    3: 'pessimism',
    4: 'guilt',
    5: 'anxiety', 
    6: 'catastrophic thinking',
    7: 'perfectionism',
    8: 'active avoidance',
    9: 'grandiosity (delusion of grandeur)', 
    10: 'manic mood',
    11: 'impulsivity',
    12: 'risk-seeking',
    13: 'splitting (binary thinking)',
    14: 'unstable self-image',
    15: 'aggression',
    16: 'anger',
    17: 'irritability'.

Dimensions 7, 13, and 14 were not used for the paper's analysis.

---

## Q&A
We provide the code to run Q&A sessions.
 - [qna_session.ipynb]: Q&A session code (See Appendix C). **Figure 3: Claim 2 Result** can be reproduced with this code.

---

## Causal inference

We provide the code to run causal inference.
 - [causal_inference.ipynb]: causal inference code (See Appendix D). **Figure 4: Claim 3 Result** can be reproduced with this code.

---

## Synthetic data generation

 - [data_generation_prompt.py]: input prompt designs to generate synthetic data used to train and evaluate S3AE.
 - [./data/qna_output.zip]: dataset of unit activations used for activation dynamics and causal analysis (_needs to be unzipped_); actual texts that LLM generated were removed due to ethical and safety concerns.

---

## Contact
For any questions, please email me at syleetolow@kaist.ac.kr! 

