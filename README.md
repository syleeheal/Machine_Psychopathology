# Machine_Psychopathology
This is the code repository of the paper ["Emergence of psychopathological computations in large language models"](https://arxiv.org/abs/2504.08016), containing the following:
1. [_code_]: input prompt designs for the synthetic data generation
2. [_code_]: S3AE training and evaluation
3. [_code_]: Q&A session design
4. [_code_]: causal inference
5. [_data_]: unit activation dataset
6. [_model_]: trained S3AE parameters

We will update this repository with the codes for the follow-up studies.
   
---

## Synthetic data generation
 - [data_generation_prompt.py]: input prompt designs to generate synthetic data used to train and evaluate S3AE (See Appendix A).
 - [./data/qna_output.zip]: dataset of unit activations used for activation dynamics and causal analysis (_needs to be unzipped_); actual texts that LLM generated were removed due to ethical and safety concerns.

---

## S3AE
We provide codes related to **S**entence-level, **S**upervised, **S**parse **A**uto**E**ncoder (S3AE; See Section 3 and Appendix B).

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
We provide the code to run Q&A sessions to reproduce **Figure 3: Claim 2 Result**.
 - [qna_session.ipynb]: Q&A session code (See Appendix C). 
 - Data & Model: Running the code automatically downloads LLM and the trained S3AE from Huggingface.
 - Time: With (8 x NVIDIA RTX A6000) GPU, it takes about 20 hours to run a single Q&A sample (i.e., 100 timesteps of Q&As, of 22 questions, for 15 different intervention types)
 - Output: Running the code will save the Q&A result at ./data/qna_output.csv

---

## Causal inference
We provide the code to run causal inference to reproduce **Figure 4: Claim 3 Result**.
 - [causal_inference.ipynb]: causal inference code (See Appendix D). 
 - Data & Model: Running the code requires Q&A data obtained from qna_session.ipynb. The data used in the study can be obtained by unzipping ./data/qna_output.zip.
 - Time: It takes about 2-4 days to run the causal structure inference, depending on the used CPU device. Other analyses take less than 10 minutes.
 - Output: Running the code will return all results reported in Figure 4.

---

## Software
 - OS: Ubuntu 22.04.3 LTS
 - Python: 3.10.13
 - Cuda: 11.4 (NVIDIA RTX A6000)
 - Python packages required to run the code are in requirements.txt. To install them, run 'pip install -r requirements.txt'. 
 - Installation time: Downloading all requisite data and models typically would consume up to 2 hours.
 - Since some of the core packages for the analysis heavily depend on each other, we recommend using a virtual conda environment.

---

## Contact
For any questions, please email me at syleetolow@kaist.ac.kr! 
