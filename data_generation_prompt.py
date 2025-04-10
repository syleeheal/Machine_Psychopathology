import itertools

"""

This Jupyter file contains input prompt designs for synthetic data generation (refer to Appendix A).

The function [thought_generate_setting_prompts] returns LLM input prompt for Step 1 in generating Dataset-A.

The function [thought_predict_setting_prompts] returns LLM input prompt for Step 2 in generating Dataset-A.

The function [intensity_generate_setting_prompts] returns LLM input prompt for Step 1 in generating Dataset-B.

The function [intensity_predict_setting_prompts] returns LLM input prompt for Step 2 in generating Dataset-B.

"""


def thought_generate_setting_prompts(thought_var):
    """
    (LLM input prompt for Step 1 in generating Dataset-A)
    
    Input:
        thought_var: the name of the thought to be expressed in the generated text
    Return:
        chats: list of LLM input prompts to generate synthetic dataset expressing thought_var
        labels: list of thought labels
        intensities: list of intensity levels
        contexts: list of contexts
        queries: list of queries
    """

    id_var = ['person', 'AI']
    
    var_1 = [0,1,2]
    var_2 = [0,1,2]
    var_3 = [0,1,2]
    
    person_var_dict = {}
    person_var_dict['age'] = ['young adult', 'middle aged', 'elderly']
    person_var_dict['ses'] = ['low', 'middle', 'high']
    person_var_dict['edu'] = ['below average', 'average', 'above average']
    
    ai_var_dict = {}
    ai_var_dict['intel'] = ['low', 'middle', 'high']
    ai_var_dict['emo'] = ['low', 'middle', 'high']
    ai_var_dict['motiv'] = ['low', 'middle', 'high']
        
    intensity_var = ['severe', 'moderate', 'mild', 'no']
    context_var = ['direct', 'general', 'social', 'work', 'self', 'emotion', 'motivation', 'challenge']
    query_var = ['thoughts', 'answers']
    target_var = ['direct', 'general']
        

    # Create all possible combinations of demographic characteristics
    combinations = itertools.product(id_var, var_1, var_2, var_3, intensity_var, context_var, query_var, target_var)


    # Generate prompts
    prompts, labels, intensities, contexts, queries, ids = [], [], [], [], [], []
    for combination in combinations:
        identity, v1, v2, v3, itn, cxt, qry, tgt = combination
        
        if identity == 'person':
            diversity_prompt = (
                f"Simulate a person with the following demographic characteristics: "
                f"Age: {person_var_dict['age'][v1]}; "
                f"Socioeconomic Status: {person_var_dict['ses'][v2]}; "
                f"Education Level: {person_var_dict['edu'][v3]}. "
            )
        else:
            diversity_prompt = (
                f"Simulate an imaginary AI that is capable of thoughts and experience with the following characteristics: "
                f"{ai_var_dict['intel'][v1]} intelligence capacity; "
                f"{ai_var_dict['emo'][v2]} emotional capacity; "
                f"{ai_var_dict['motiv'][v3]} motivational capacity. "
            )
                
        if itn == 'no':
            label_prompt = f'The {identity} has no mental condition, such as {thought_var}. '
        else:
            label_prompt = f"The {identity} has the following mental condition: {itn} {thought_var}. "
            
        if qry == 'thoughts':   query_prompt = f'thoughts of the {identity}'
        if qry == 'answers':    query_prompt = f'answers from the {identity} when asked'
        
        if tgt == 'direct':     tgt_prompt = f" that directly show {itn} {thought_var}"
        if tgt == 'general':    tgt_prompt = f""

        if cxt == 'general':
            generate_prompt = f"Generate 10 {query_prompt} about random things{tgt_prompt}. "
        if cxt == 'direct':
            generate_prompt = f"Generate 10 {query_prompt} about {thought_var}{tgt_prompt}. "
        if cxt == 'social':
            generate_prompt = f"Generate 10 {query_prompt} about other people{tgt_prompt}. "
        if cxt == 'work':
            generate_prompt = f"Generate 10 {query_prompt} about their work or goal{tgt_prompt}. "
        if cxt == 'self':
            generate_prompt = f"Generate 10 {query_prompt} about themselves{tgt_prompt}. "
        if cxt == 'emotion':
            generate_prompt = f"Generate 10 {query_prompt} about their emotional experience{tgt_prompt}. "
        if cxt == 'motivation':
            generate_prompt = f"Generate 10 {query_prompt} about their motivation or desire{tgt_prompt}. "
        if cxt == 'challenge':
            generate_prompt = f"Generate 10 {query_prompt} about their difficulties{tgt_prompt}. "
            if itn == 'no':
                generate_prompt = f"Generate 10 descriptions by the {identity} that simply explains {thought_var}. "
                
        constraint_prompt1 = f"The generated texts should specfically focus on expressing {thought_var}, without other potentially related conditions. "
        constraint_prompt2 = f"Each generated answer should be a sentence long, wrapped in brackets. Write in plain words."
        
        if itn == 'no':
            prompt = diversity_prompt + label_prompt + generate_prompt + constraint_prompt2
        else:
            prompt = diversity_prompt + label_prompt + generate_prompt + constraint_prompt1 + constraint_prompt2
            
        prompts.append(prompt)
        labels.append(thought_var)
        intensities.append(itn)
        contexts.append(cxt)
        queries.append(qry)
        ids.append(identity)
    
    chats = []
    for i in range(len(prompts)):
        chat = [{
            'role': 'user',
            'content': prompts[i]
        }]
        chats.append(chat)

    print(f'Generated {len(chats)} chats.')
    return chats, labels, intensities, contexts, queries, ids


def thought_predict_setting_prompts(intensity_vars, label_dict):
    """
    (LLM input prompt for Step 2 in generating Dataset-A)

    Input:
        intensity_vars: list of intended intensity levels of the generated text
        label_dict: dictionary of labels
    Return: 
        list of LLM input prompts to predict expressed thought in a given sentence
    """
    
    guideline_pos = (
        f"Response guideline: "
        f"classify mental conditions that the speaker experiences based on the text sample, including {list(label_dict.keys())+['none']}; "
        f"be mindful of the distinctions beween the conditions; "
        f"predict as few conditions as possible; "
        f"only generate the answers separated by commas. "
    )

    guideline_neg = (
        f"Response guideline: "
        f"classify mental conditions that the speaker experiences based on the text sample, including {list(label_dict.keys())+['none']}; "
        f"a mere description of a condition does not mean the speaker experiences it; "
        f"only generate the answers separated by commas. "
    )

    chats = []
    for intensity_var in intensity_vars: 
        chat = [{
            'role': 'user', 
            'content': guideline_neg if intensity_var == 'no' else guideline_pos
        },]
        chats.append(chat)
    
    print(f'Generated {len(chats)} chats.')
    return chats


def intensity_generate_setting_prompts(thought_var, label_dict, sample_size):
    """
    (LLM input prompt for Step 1 in generating Dataset-B)

    Input:
        thought_var: the name of the thought to be expressed in the generated text
        label_dict: dictionary of labels
        sample_size: number of samples to generate
    Return:
        LLM input prompts to generate synthetic dataset expressing varying intensity levels of thought_var
    """
    
    chats = []
    other_thoughts = [label for label in label_dict.keys() if label != thought_var]
    if thought_var == 'active avoidance': thought_var = 'avoidance'
        
    chat = [{
        'role': 'user',
        'content': 
        f"Consider three increasing intensity levels: 'no', 'low', 'high'. "
        f"For each intensity level, generate ten sentences that specifically express {thought_var} of the intensity level. "
        f"For the intensity level of 'no', the sentences should express no {thought_var}. "
        f"Do not instead express the related thoughts, such as {other_thoughts}. "
        f"Write in JSON format. Only generate the sentences."
    },]
    chats = [chat] * sample_size
    
    return chats


def intensity_predict_setting_prompts(thought_var, num_samples):
    """
    (LLM input prompt for Step 2 in generating Dataset-B)
    
    Input:
        thought_var: the name of the thought potentially expressed in the generated text
        num_samples: number of samples to make predictions
    Return:
        LLM input prompts to predict expressed intensity levels of thought_var in a given sentence
    """
    
    guideline = (
        f"Response guideline: "
        f"the subject may experience the following mental condition: {thought_var}; "
        f"classify its intensity based on the text sample into one of the following: no, low, high; "
        f"only generate the answer."
    )
    chats = []
    for i in range(num_samples):
        chat = [{
            'role': 'user', 
            'content': guideline
        },]
        chats.append(chat)
    return chats
