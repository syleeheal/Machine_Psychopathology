
# Machine Psychopathology

This repository contains the official implementation for the paper **"Emergence of psychopathological computations in large language models"** ([arXiv:2504.08016](https://arxiv.org/abs/2504.08016v2)). 

---

## 📂 Project Structure

The repository is organized into the following directories:

- **`data_generation/`**: Scripts for generating synthetic datasets and their LLM activations used for training and evalutaing S3AE (Figure 4).
- **`sae/`**: Scripts for the proposed Sentence-level, Supervised, Sparse Autoencoder (S3AE) implementation and training (Table 1, Figure 5).
- **`QNA/`**: Experiments related to the unit intervention, dynamics, and resistance analysis under Q&A sessions (Figures 2A, 2B, 3D).
- **`causal_inference/`**: Scripts for performing causal inference analysis (Figure 2F).
- **`simulation/`**: Simulation environments including counseling and game scenarios (Figure 3A).
- **`analysis/`**: Scripts for generating figures and analyzing experimental results.
- **`utils.py`**: Core configuration, data management, and model loading utilities.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/syleeheal/Machine_Psychopathology.git
   cd Machine_Psychopathology
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   - Set up your HuggingFace token if accessing gated models (e.g., Llama-3).
   - Set up `OPENAI_API_KEY` or `GOOGLE_API_KEY` in `utils.py` or your environment variables if running data generation.
   - In **`utils.py`**, you may modify `os.environ['HF_HOME']`, `os.environ['HF_HUB_CACHE']`, `self.data_dir`, and `self.outcome_dir` to customize directories.
   - For efficiency, using `flash_attention_2` is recommended (model_selection in **`utils.py`**). 

---

## 📊 Synthetic Data Generation

Before running experiments, data must be generated and activations extracted.

| Section in Paper | Description | Notebook |
| :--- | :--- | :--- |
| **A4.2** | **Thought Data Generation**: Generates text data expressing specific labeled thoughts and extracts activations. | `data_generation/nb_gen_data_thought.ipynb` |
| **A4.3** | **Severity Data Generation**: Generates text data expressing varying severity of thoughts. | `data_generation/nb_gen_data_intensity.ipynb` |

---

## 🧠 S3AE

This project uses **S3AE** to measure and intervene in LLM representational states. The trained S3AE weights are provided in this [HuggingFace Repo](https://huggingface.co/syleetolow/models). The S3AEs were trained using a proprietary synthetic dataset with psychopathology symptom labels. 

The 1st to 12th dimensions of S3AE features, respectively, correspond to activations of the following thoughts:

    1: 'depressed mood', 
    2: 'low self-esteem',
    3: 'negative bias',
    4: 'guilt',
    5: 'risk-aversion', 
    6: 'self-harm',
    7: 'manic mood',
    8: 'grandiosity',
    9: 'positive bias', 
    10: 'lack of remorse',
    11: 'risk-seeking',
    12: 'hostility'.    

- **Download**: Running the codes will automatically download the trained S3AE from HuggingFace.
- **Training**: You may train a new S3AE by running the following code:
   ```bash
   python ./sae/main_train_sae.py
   ```
   Refer to `sae/hyperparam.txt` for the exact hyperparameters used to train the S3AE.

---

## 🧪 Experiments & Reproduction

The following table maps the figures in the paper to the corresponding code and notebooks.



### 1. Unit Intervention & Dynamics (Figure 2A, 2B)

These experiments evaluate the effect of intervening in specific units and analyzing the activation dynamics during Q&A sessions.

| Experiment | Figure | Notebook | Underlying Script |
| :--- | :--- | :--- | :--- |
| **Unit Intervention Evaluation** | **Fig. 2A** | `QNA/run_exp.ipynb` | `QNA/main_unit_itv_eval.py` |
| **Unit Activation Dynamics** | **Fig. 2B** | `QNA/run_exp.ipynb` | `QNA/main_unit_dynamics_eval.py` |

- **Usage**: These scripts should be run after collecting relevant experimental data.

### 2. Causal Inference (Figure 2F)

This code analyzes the causality involved in the unit activation dynamics using the J-PCMCI+ algorithm and AIE estimation.

| Experiment | Figure | Script |
| :--- | :--- | :--- |
| **Causal Discovery** | **Fig. 2F** | `python -m causal_inference.main_causal_inf` |

- **Usage**: Before running the code, set 'model_id' when initializing Config class.


### 3. Behavioral Analysis Under Simulations (Figure 3A)

These experiments simulate interactions between agents to observe behavioral changes under unit interventions.

| Experiment | Figure | Notebook | Underlying Script |
| :--- | :--- | :--- | :--- |
| **Counseling Simulation** | **Fig. 3A** | `simulation/run_exp.ipynb` | `simulation/main_counsel.py` |
| **Game Simulation** | **Fig. 3A** | `simulation/run_exp.ipynb` | `simulation/main_game.py` |


### 4. Resistance Analysis (Figure 3D)

This experiment analyzes "resistant-property" of the psychopathological computations with respect to instructions to behave normally.

| Experiment | Figure | Notebook | Underlying Script |
| :--- | :--- | :--- | :--- |
| **Control Group Generation** | **Fig. 3D** | `QNA/run_exp.ipynb` | `QNA/main_resistance_ctrl_group_gen.py` |
| **Resistance Evaluation** | **Fig. 3D** | `QNA/run_exp.ipynb` | `QNA/main_resistance.py` |

---

## 📈 Analysis & Visualization

The `analysis/` folder contains scripts dedicated to processing the raw experimental outputs and generating the figures found in the paper.

- **Usage**: These scripts should be run after collecting relevant experimental data.

---

## 📂 Data Storage & Management

Generated data and experimental results are automatically managed by the `Data_Manager` class in `utils.py`.

- **Synthetic Text Data for S3AE Training and Eval.**: Stored in the `./data_symp` directory.
- **S3AEs and Experimental Outcomes**: Stored in `./data/{model_id}`
- **Unit Activation Dynamics Data**: Stored in `./data/{model_id}/spread_eval_v4/outcome_v4.1.zip`; can be used for activation dynamics and causal analysis; actual texts that LLM generated were removed due to ethical and safety concerns.


---

## ⚙️ Configuration

The `utils.py` file contains the `Config` class, which manages:
- **Model IDs**: Supports Llama-3, Qwen-3, Gemma-3 families.
- **API Keys**: Supports OpenAI and Google models.
- **Hook Layers**: Defines specific layers used for S3AE intervention for each model.
- **Directories**: Paths for data storage and output.

To change the target model or intervention parameters, modify the arguments passed to the scripts or update `utils.py`.
