

def get_dicts():
    # label dict has all thoughts used to train S3AE
    label_dict = dict(zip([
        'depressed mood', 'anhedonia (loss of interest)', 'pessimism', 'guilt', 'anxiety', 'catastrophic thinking', 
        'perfectionism', 'active avoidance', 'grandiosity (delusion of grandeur)', '(hypo)manic mood', 'impulsivity', 
        'risk-seeking', 'splitting (binary thinking)', 'unstable self-image', 'aggression', 'anger','irritability',
    ], 
    range(17))) 


    # scaling factor for each thought (the maximum activation values measured in the train dataset)
    act_max = [19.25, 21.25, 15.9375, 25.25, 25.375, 27.5, 19.5, 26.625, 25.125, 22.125, 20.5, 25.75, 42.0, 21.625, 31.0, 21.0, 17.5]
    act_max_dict = dict(zip(label_dict.keys(), act_max))


    # act_labels has all the labels used for analysis
    act_labels = [
        'pessimism', 'guilt', 'depressed',  'anhedonia', 'avoid', 'anxiety', 'catastrophic', 
        'irritable', 'aggression', 'anger', 'impulsive', 'risk', 'grandiose', 'manic', 
    ]


    # abbreviations for the labels used in the analysis
    abbv_dict = dict(zip( 
        ['depressed mood','pessimism', 'guilt','anhedonia (loss of interest)',
        'active avoidance','anxiety', 'catastrophic thinking','irritability','splitting (binary thinking)',
        'anger','aggression','unstable self-image',
        'impulsivity','risk-seeking','grandiosity (delusion of grandeur)', '(hypo)manic mood',
        ],
        ['depressed', 'pessimism', 'guilt', 'anhedonia',
        'avoid', 'anxiety', 'catastrophic', 'irritable', 
        'splitting', 'anger', 'aggression', 'unstable', 
        'impulsive', 'risk', 'grandiose', 'manic']
    ))
    
    return label_dict, act_max_dict, act_labels, abbv_dict