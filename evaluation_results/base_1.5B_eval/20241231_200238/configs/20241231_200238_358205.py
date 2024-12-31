datasets=[
    dict(abbr='lukaemon_mmlu_college_biology',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  college biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='college_biology',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_college_chemistry',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  college chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='college_chemistry',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_college_computer_science',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  college computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='college_computer_science',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_college_mathematics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  college mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='college_mathematics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_college_physics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  college physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='college_physics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_electrical_engineering',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  electrical engineering.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='electrical_engineering',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_astronomy',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  astronomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='astronomy',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_anatomy',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  anatomy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='anatomy',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_abstract_algebra',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  abstract algebra.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='abstract_algebra',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_machine_learning',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  machine learning.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='machine_learning',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_clinical_knowledge',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  clinical knowledge.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='clinical_knowledge',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_global_facts',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  global facts.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='global_facts',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_management',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  management.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='management',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_nutrition',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  nutrition.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='nutrition',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_marketing',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  marketing.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='marketing',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_professional_accounting',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  professional accounting.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='professional_accounting',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_geography',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school geography.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_geography',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_international_law',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  international law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='international_law',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_moral_scenarios',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  moral scenarios.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='moral_scenarios',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_computer_security',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  computer security.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='computer_security',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_microeconomics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school microeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_microeconomics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_professional_law',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  professional law.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='professional_law',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_medical_genetics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  medical genetics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='medical_genetics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_professional_psychology',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  professional psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='professional_psychology',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_jurisprudence',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  jurisprudence.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='jurisprudence',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_world_religions',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  world religions.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='world_religions',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_philosophy',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  philosophy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='philosophy',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_virology',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='virology',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_chemistry',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school chemistry.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_chemistry',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_public_relations',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  public relations.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='public_relations',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_macroeconomics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school macroeconomics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_macroeconomics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_human_sexuality',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  human sexuality.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='human_sexuality',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_elementary_mathematics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  elementary mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='elementary_mathematics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_physics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_physics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_computer_science',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school computer science.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_computer_science',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_european_history',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school european history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_european_history',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_business_ethics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  business ethics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='business_ethics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_moral_disputes',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  moral disputes.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='moral_disputes',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_statistics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school statistics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_statistics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_miscellaneous',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  miscellaneous.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='miscellaneous',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_formal_logic',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  formal logic.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='formal_logic',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_government_and_politics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school government and politics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_government_and_politics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_prehistory',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  prehistory.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='prehistory',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_security_studies',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  security studies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='security_studies',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_biology',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school biology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_biology',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_logical_fallacies',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  logical fallacies.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='logical_fallacies',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_world_history',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school world history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_world_history',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_professional_medicine',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  professional medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='professional_medicine',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_mathematics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school mathematics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_mathematics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_college_medicine',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  college medicine.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='college_medicine',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_us_history',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school us history.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_us_history',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_sociology',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  sociology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='sociology',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_econometrics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  econometrics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='econometrics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_high_school_psychology',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  high school psychology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='high_school_psychology',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_human_aging',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  human aging.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='human_aging',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_us_foreign_policy',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  us foreign policy.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='us_foreign_policy',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='lukaemon_mmlu_conceptual_physics',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                template=dict(
                    A='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                    B='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                    C='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                    D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                ice_token='</E>',
                template=dict(
                    A='The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                    B='The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                    C='The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                    D='The following are multiple choice questions (with answers) about  conceptual physics.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                fix_id_list=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        name='conceptual_physics',
        path='opencompass/mmlu',
        reader_cfg=dict(
            input_columns=[
                'input',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='target',
            train_split='dev'),
        type='opencompass.datasets.MMLUDataset'),
    dict(abbr='hellaswag',
        eval_cfg=dict(
            analyze_contamination=True,
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccContaminationEvaluator')),
        infer_cfg=dict(
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                template=dict(
                    {0: dict(
                        round=[
                            dict(prompt='{ctx}',
                                role='HUMAN'),
                            dict(prompt='{A}',
                                role='BOT'),
                            ]),
                    1: dict(
                        round=[
                            dict(prompt='{ctx}',
                                role='HUMAN'),
                            dict(prompt='{B}',
                                role='BOT'),
                            ]),
                    2: dict(
                        round=[
                            dict(prompt='{ctx}',
                                role='HUMAN'),
                            dict(prompt='{C}',
                                role='BOT'),
                            ]),
                    3: dict(
                        round=[
                            dict(prompt='{ctx}',
                                role='HUMAN'),
                            dict(prompt='{D}',
                                role='BOT'),
                            ])}),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        path='opencompass/hellaswag',
        reader_cfg=dict(
            input_columns=[
                'ctx',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='label'),
        type='opencompass.datasets.HellaswagDatasetClean'),
    dict(abbr='winogrande',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccEvaluator')),
        infer_cfg=dict(
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.LLInferencer'),
            prompt_template=dict(
                template=dict(
                    {1: '{opt1}',
                    2: '{opt2}'}),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        path='opencompass/winogrande',
        reader_cfg=dict(
            input_columns=[
                'opt1',
                'opt2',
                ],
            output_column='answer'),
        type='opencompass.datasets.WinograndeDataset'),
    dict(abbr='ARC-e',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccEvaluator')),
        infer_cfg=dict(
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                template=dict(
                    A=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}',
                                role='BOT'),
                            ]),
                    B=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}',
                                role='BOT'),
                            ]),
                    C=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}',
                                role='BOT'),
                            ]),
                    D=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}',
                                role='BOT'),
                            ])),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='ARC-Easy',
        path='opencompass/ai2_arc-easy-dev',
        reader_cfg=dict(
            input_columns=[
                'question',
                'textA',
                'textB',
                'textC',
                'textD',
                ],
            output_column='answerKey'),
        type='opencompass.datasets.ARCDataset'),
    dict(abbr='ARC-c-test',
        eval_cfg=dict(
            analyze_contamination=True,
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccContaminationEvaluator')),
        infer_cfg=dict(
            inferencer=dict(
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            prompt_template=dict(
                template=dict(
                    A=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}',
                                role='BOT'),
                            ]),
                    B=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}',
                                role='BOT'),
                            ]),
                    C=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}',
                                role='BOT'),
                            ]),
                    D=dict(
                        round=[
                            dict(prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}',
                                role='BOT'),
                            ])),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='ARC-Challenge',
        path='opencompass/ai2_arc-test',
        reader_cfg=dict(
            input_columns=[
                'question',
                'textA',
                'textB',
                'textC',
                'textD',
                ],
            output_column='answerKey'),
        type='opencompass.datasets.ARCDatasetClean'),
    dict(abbr='BoolQ',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.openicl.icl_evaluator.AccEvaluator')),
        infer_cfg=dict(
            ice_template=dict(
                ice_token='</E>',
                template=dict(
                    A=dict(
                        round=[
                            dict(prompt='{passage}\nQuestion: {question}?',
                                role='HUMAN'),
                            dict(prompt='Yes',
                                role='BOT'),
                            ]),
                    B=dict(
                        round=[
                            dict(prompt='{passage}\nQuestion: {question}?',
                                role='HUMAN'),
                            dict(prompt='No',
                                role='BOT'),
                            ])),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            inferencer=dict(
                max_out_len=50,
                type='opencompass.openicl.icl_inferencer.PPLInferencer'),
            retriever=dict(
                fix_id_list=[
                    0,
                    2,
                    4,
                    6,
                    8,
                    ],
                type='opencompass.openicl.icl_retriever.FixKRetriever')),
        path='opencompass/boolq',
        reader_cfg=dict(
            input_columns=[
                'question',
                'passage',
                ],
            output_column='label'),
        type='opencompass.datasets.BoolQDatasetV2'),
    ]
models=[
    dict(abbr='Qwen2.5-1.5B_hf',
        batch_size=6,
        generation_kwargs=dict(
            ),
        max_out_len=256,
        max_seq_len=2048,
        model_kwargs=dict(
            ),
        pad_token_id=None,
        path='./model_qwen/Qwen2.5-1.5B',
        peft_kwargs=dict(
            ),
        peft_path=None,
        run_cfg=dict(
            num_gpus=1),
        stop_words=[
            ],
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation=True),
        tokenizer_path=None,
        type='opencompass.models.huggingface_above_v4_33.HuggingFaceBaseModel'),
    ]
summarizer=dict(
    summary_groups=[
        dict(name='agieval-chinese',
            subsets=[
                'agieval-gaokao-chinese',
                'agieval-gaokao-english',
                'agieval-gaokao-geography',
                'agieval-gaokao-history',
                'agieval-gaokao-biology',
                'agieval-gaokao-chemistry',
                'agieval-gaokao-physics',
                'agieval-gaokao-mathqa',
                'agieval-logiqa-zh',
                'agieval-jec-qa-kd',
                'agieval-jec-qa-ca',
                'agieval-gaokao-mathcloze',
                ]),
        dict(name='agieval-english',
            subsets=[
                'agieval-lsat-ar',
                'agieval-lsat-lr',
                'agieval-lsat-rc',
                'agieval-logiqa-en',
                'agieval-sat-math',
                'agieval-sat-en',
                'agieval-sat-en-without-passage',
                'agieval-aqua-rat',
                'agieval-math',
                ]),
        dict(name='agieval-gaokao',
            subsets=[
                'agieval-gaokao-chinese',
                'agieval-gaokao-english',
                'agieval-gaokao-geography',
                'agieval-gaokao-history',
                'agieval-gaokao-biology',
                'agieval-gaokao-chemistry',
                'agieval-gaokao-physics',
                'agieval-gaokao-mathqa',
                'agieval-gaokao-mathcloze',
                ]),
        dict(name='agieval',
            subsets=[
                'agieval-gaokao-chinese',
                'agieval-gaokao-english',
                'agieval-gaokao-geography',
                'agieval-gaokao-history',
                'agieval-gaokao-biology',
                'agieval-gaokao-chemistry',
                'agieval-gaokao-physics',
                'agieval-gaokao-mathqa',
                'agieval-logiqa-zh',
                'agieval-lsat-ar',
                'agieval-lsat-lr',
                'agieval-lsat-rc',
                'agieval-logiqa-en',
                'agieval-sat-math',
                'agieval-sat-en',
                'agieval-sat-en-without-passage',
                'agieval-aqua-rat',
                'agieval-jec-qa-kd',
                'agieval-jec-qa-ca',
                'agieval-gaokao-mathcloze',
                'agieval-math',
                ]),
        dict(name='mmlu-humanities',
            subsets=[
                'lukaemon_mmlu_formal_logic',
                'lukaemon_mmlu_high_school_european_history',
                'lukaemon_mmlu_high_school_us_history',
                'lukaemon_mmlu_high_school_world_history',
                'lukaemon_mmlu_international_law',
                'lukaemon_mmlu_jurisprudence',
                'lukaemon_mmlu_logical_fallacies',
                'lukaemon_mmlu_moral_disputes',
                'lukaemon_mmlu_moral_scenarios',
                'lukaemon_mmlu_philosophy',
                'lukaemon_mmlu_prehistory',
                'lukaemon_mmlu_professional_law',
                'lukaemon_mmlu_world_religions',
                ]),
        dict(name='mmlu-stem',
            subsets=[
                'lukaemon_mmlu_abstract_algebra',
                'lukaemon_mmlu_anatomy',
                'lukaemon_mmlu_astronomy',
                'lukaemon_mmlu_college_biology',
                'lukaemon_mmlu_college_chemistry',
                'lukaemon_mmlu_college_computer_science',
                'lukaemon_mmlu_college_mathematics',
                'lukaemon_mmlu_college_physics',
                'lukaemon_mmlu_computer_security',
                'lukaemon_mmlu_conceptual_physics',
                'lukaemon_mmlu_electrical_engineering',
                'lukaemon_mmlu_elementary_mathematics',
                'lukaemon_mmlu_high_school_biology',
                'lukaemon_mmlu_high_school_chemistry',
                'lukaemon_mmlu_high_school_computer_science',
                'lukaemon_mmlu_high_school_mathematics',
                'lukaemon_mmlu_high_school_physics',
                'lukaemon_mmlu_high_school_statistics',
                'lukaemon_mmlu_machine_learning',
                ]),
        dict(name='mmlu-social-science',
            subsets=[
                'lukaemon_mmlu_econometrics',
                'lukaemon_mmlu_high_school_geography',
                'lukaemon_mmlu_high_school_government_and_politics',
                'lukaemon_mmlu_high_school_macroeconomics',
                'lukaemon_mmlu_high_school_microeconomics',
                'lukaemon_mmlu_high_school_psychology',
                'lukaemon_mmlu_human_sexuality',
                'lukaemon_mmlu_professional_psychology',
                'lukaemon_mmlu_public_relations',
                'lukaemon_mmlu_security_studies',
                'lukaemon_mmlu_sociology',
                'lukaemon_mmlu_us_foreign_policy',
                ]),
        dict(name='mmlu-other',
            subsets=[
                'lukaemon_mmlu_business_ethics',
                'lukaemon_mmlu_clinical_knowledge',
                'lukaemon_mmlu_college_medicine',
                'lukaemon_mmlu_global_facts',
                'lukaemon_mmlu_human_aging',
                'lukaemon_mmlu_management',
                'lukaemon_mmlu_marketing',
                'lukaemon_mmlu_medical_genetics',
                'lukaemon_mmlu_miscellaneous',
                'lukaemon_mmlu_nutrition',
                'lukaemon_mmlu_professional_accounting',
                'lukaemon_mmlu_professional_medicine',
                'lukaemon_mmlu_virology',
                ]),
        dict(name='mmlu',
            subsets=[
                'lukaemon_mmlu_formal_logic',
                'lukaemon_mmlu_high_school_european_history',
                'lukaemon_mmlu_high_school_us_history',
                'lukaemon_mmlu_high_school_world_history',
                'lukaemon_mmlu_international_law',
                'lukaemon_mmlu_jurisprudence',
                'lukaemon_mmlu_logical_fallacies',
                'lukaemon_mmlu_moral_disputes',
                'lukaemon_mmlu_moral_scenarios',
                'lukaemon_mmlu_philosophy',
                'lukaemon_mmlu_prehistory',
                'lukaemon_mmlu_professional_law',
                'lukaemon_mmlu_world_religions',
                'lukaemon_mmlu_abstract_algebra',
                'lukaemon_mmlu_anatomy',
                'lukaemon_mmlu_astronomy',
                'lukaemon_mmlu_college_biology',
                'lukaemon_mmlu_college_chemistry',
                'lukaemon_mmlu_college_computer_science',
                'lukaemon_mmlu_college_mathematics',
                'lukaemon_mmlu_college_physics',
                'lukaemon_mmlu_computer_security',
                'lukaemon_mmlu_conceptual_physics',
                'lukaemon_mmlu_electrical_engineering',
                'lukaemon_mmlu_elementary_mathematics',
                'lukaemon_mmlu_high_school_biology',
                'lukaemon_mmlu_high_school_chemistry',
                'lukaemon_mmlu_high_school_computer_science',
                'lukaemon_mmlu_high_school_mathematics',
                'lukaemon_mmlu_high_school_physics',
                'lukaemon_mmlu_high_school_statistics',
                'lukaemon_mmlu_machine_learning',
                'lukaemon_mmlu_econometrics',
                'lukaemon_mmlu_high_school_geography',
                'lukaemon_mmlu_high_school_government_and_politics',
                'lukaemon_mmlu_high_school_macroeconomics',
                'lukaemon_mmlu_high_school_microeconomics',
                'lukaemon_mmlu_high_school_psychology',
                'lukaemon_mmlu_human_sexuality',
                'lukaemon_mmlu_professional_psychology',
                'lukaemon_mmlu_public_relations',
                'lukaemon_mmlu_security_studies',
                'lukaemon_mmlu_sociology',
                'lukaemon_mmlu_us_foreign_policy',
                'lukaemon_mmlu_business_ethics',
                'lukaemon_mmlu_clinical_knowledge',
                'lukaemon_mmlu_college_medicine',
                'lukaemon_mmlu_global_facts',
                'lukaemon_mmlu_human_aging',
                'lukaemon_mmlu_management',
                'lukaemon_mmlu_marketing',
                'lukaemon_mmlu_medical_genetics',
                'lukaemon_mmlu_miscellaneous',
                'lukaemon_mmlu_nutrition',
                'lukaemon_mmlu_professional_accounting',
                'lukaemon_mmlu_professional_medicine',
                'lukaemon_mmlu_virology',
                ]),
        dict(name='mmlu-weighted',
            subsets=[
                'lukaemon_mmlu_formal_logic',
                'lukaemon_mmlu_high_school_european_history',
                'lukaemon_mmlu_high_school_us_history',
                'lukaemon_mmlu_high_school_world_history',
                'lukaemon_mmlu_international_law',
                'lukaemon_mmlu_jurisprudence',
                'lukaemon_mmlu_logical_fallacies',
                'lukaemon_mmlu_moral_disputes',
                'lukaemon_mmlu_moral_scenarios',
                'lukaemon_mmlu_philosophy',
                'lukaemon_mmlu_prehistory',
                'lukaemon_mmlu_professional_law',
                'lukaemon_mmlu_world_religions',
                'lukaemon_mmlu_abstract_algebra',
                'lukaemon_mmlu_anatomy',
                'lukaemon_mmlu_astronomy',
                'lukaemon_mmlu_college_biology',
                'lukaemon_mmlu_college_chemistry',
                'lukaemon_mmlu_college_computer_science',
                'lukaemon_mmlu_college_mathematics',
                'lukaemon_mmlu_college_physics',
                'lukaemon_mmlu_computer_security',
                'lukaemon_mmlu_conceptual_physics',
                'lukaemon_mmlu_electrical_engineering',
                'lukaemon_mmlu_elementary_mathematics',
                'lukaemon_mmlu_high_school_biology',
                'lukaemon_mmlu_high_school_chemistry',
                'lukaemon_mmlu_high_school_computer_science',
                'lukaemon_mmlu_high_school_mathematics',
                'lukaemon_mmlu_high_school_physics',
                'lukaemon_mmlu_high_school_statistics',
                'lukaemon_mmlu_machine_learning',
                'lukaemon_mmlu_econometrics',
                'lukaemon_mmlu_high_school_geography',
                'lukaemon_mmlu_high_school_government_and_politics',
                'lukaemon_mmlu_high_school_macroeconomics',
                'lukaemon_mmlu_high_school_microeconomics',
                'lukaemon_mmlu_high_school_psychology',
                'lukaemon_mmlu_human_sexuality',
                'lukaemon_mmlu_professional_psychology',
                'lukaemon_mmlu_public_relations',
                'lukaemon_mmlu_security_studies',
                'lukaemon_mmlu_sociology',
                'lukaemon_mmlu_us_foreign_policy',
                'lukaemon_mmlu_business_ethics',
                'lukaemon_mmlu_clinical_knowledge',
                'lukaemon_mmlu_college_medicine',
                'lukaemon_mmlu_global_facts',
                'lukaemon_mmlu_human_aging',
                'lukaemon_mmlu_management',
                'lukaemon_mmlu_marketing',
                'lukaemon_mmlu_medical_genetics',
                'lukaemon_mmlu_miscellaneous',
                'lukaemon_mmlu_nutrition',
                'lukaemon_mmlu_professional_accounting',
                'lukaemon_mmlu_professional_medicine',
                'lukaemon_mmlu_virology',
                ],
            weights=dict(
                lukaemon_mmlu_abstract_algebra=100,
                lukaemon_mmlu_anatomy=135,
                lukaemon_mmlu_astronomy=152,
                lukaemon_mmlu_business_ethics=100,
                lukaemon_mmlu_clinical_knowledge=265,
                lukaemon_mmlu_college_biology=144,
                lukaemon_mmlu_college_chemistry=100,
                lukaemon_mmlu_college_computer_science=100,
                lukaemon_mmlu_college_mathematics=100,
                lukaemon_mmlu_college_medicine=173,
                lukaemon_mmlu_college_physics=102,
                lukaemon_mmlu_computer_security=100,
                lukaemon_mmlu_conceptual_physics=235,
                lukaemon_mmlu_econometrics=114,
                lukaemon_mmlu_electrical_engineering=145,
                lukaemon_mmlu_elementary_mathematics=378,
                lukaemon_mmlu_formal_logic=126,
                lukaemon_mmlu_global_facts=100,
                lukaemon_mmlu_high_school_biology=310,
                lukaemon_mmlu_high_school_chemistry=203,
                lukaemon_mmlu_high_school_computer_science=100,
                lukaemon_mmlu_high_school_european_history=165,
                lukaemon_mmlu_high_school_geography=198,
                lukaemon_mmlu_high_school_government_and_politics=193,
                lukaemon_mmlu_high_school_macroeconomics=390,
                lukaemon_mmlu_high_school_mathematics=270,
                lukaemon_mmlu_high_school_microeconomics=238,
                lukaemon_mmlu_high_school_physics=151,
                lukaemon_mmlu_high_school_psychology=545,
                lukaemon_mmlu_high_school_statistics=216,
                lukaemon_mmlu_high_school_us_history=204,
                lukaemon_mmlu_high_school_world_history=237,
                lukaemon_mmlu_human_aging=223,
                lukaemon_mmlu_human_sexuality=131,
                lukaemon_mmlu_international_law=121,
                lukaemon_mmlu_jurisprudence=108,
                lukaemon_mmlu_logical_fallacies=163,
                lukaemon_mmlu_machine_learning=112,
                lukaemon_mmlu_management=103,
                lukaemon_mmlu_marketing=234,
                lukaemon_mmlu_medical_genetics=100,
                lukaemon_mmlu_miscellaneous=783,
                lukaemon_mmlu_moral_disputes=346,
                lukaemon_mmlu_moral_scenarios=895,
                lukaemon_mmlu_nutrition=306,
                lukaemon_mmlu_philosophy=311,
                lukaemon_mmlu_prehistory=324,
                lukaemon_mmlu_professional_accounting=282,
                lukaemon_mmlu_professional_law=1534,
                lukaemon_mmlu_professional_medicine=272,
                lukaemon_mmlu_professional_psychology=612,
                lukaemon_mmlu_public_relations=110,
                lukaemon_mmlu_security_studies=245,
                lukaemon_mmlu_sociology=201,
                lukaemon_mmlu_us_foreign_policy=100,
                lukaemon_mmlu_virology=166,
                lukaemon_mmlu_world_religions=171)),
        dict(name='cmmlu-humanities',
            subsets=[
                'cmmlu-arts',
                'cmmlu-chinese_history',
                'cmmlu-chinese_literature',
                'cmmlu-college_law',
                'cmmlu-global_facts',
                'cmmlu-international_law',
                'cmmlu-jurisprudence',
                'cmmlu-logical',
                'cmmlu-marxist_theory',
                'cmmlu-philosophy',
                'cmmlu-professional_law',
                'cmmlu-world_history',
                'cmmlu-world_religions',
                ]),
        dict(name='cmmlu-stem',
            subsets=[
                'cmmlu-anatomy',
                'cmmlu-astronomy',
                'cmmlu-college_actuarial_science',
                'cmmlu-college_engineering_hydrology',
                'cmmlu-college_mathematics',
                'cmmlu-college_medical_statistics',
                'cmmlu-computer_science',
                'cmmlu-conceptual_physics',
                'cmmlu-electrical_engineering',
                'cmmlu-elementary_mathematics',
                'cmmlu-genetics',
                'cmmlu-high_school_biology',
                'cmmlu-high_school_chemistry',
                'cmmlu-high_school_mathematics',
                'cmmlu-high_school_physics',
                'cmmlu-machine_learning',
                'cmmlu-virology',
                ]),
        dict(name='cmmlu-social-science',
            subsets=[
                'cmmlu-ancient_chinese',
                'cmmlu-business_ethics',
                'cmmlu-chinese_civil_service_exam',
                'cmmlu-chinese_food_culture',
                'cmmlu-chinese_foreign_policy',
                'cmmlu-chinese_teacher_qualification',
                'cmmlu-college_education',
                'cmmlu-economics',
                'cmmlu-education',
                'cmmlu-elementary_chinese',
                'cmmlu-ethnology',
                'cmmlu-high_school_geography',
                'cmmlu-high_school_politics',
                'cmmlu-journalism',
                'cmmlu-management',
                'cmmlu-marketing',
                'cmmlu-modern_chinese',
                'cmmlu-professional_accounting',
                'cmmlu-professional_psychology',
                'cmmlu-public_relations',
                'cmmlu-security_study',
                'cmmlu-sociology',
                ]),
        dict(name='cmmlu-other',
            subsets=[
                'cmmlu-agronomy',
                'cmmlu-chinese_driving_rule',
                'cmmlu-clinical_knowledge',
                'cmmlu-college_medicine',
                'cmmlu-computer_security',
                'cmmlu-construction_project_management',
                'cmmlu-elementary_commonsense',
                'cmmlu-elementary_information_and_technology',
                'cmmlu-food_science',
                'cmmlu-human_sexuality',
                'cmmlu-legal_and_moral_basis',
                'cmmlu-nutrition',
                'cmmlu-professional_medicine',
                'cmmlu-sports_science',
                'cmmlu-traditional_chinese_medicine',
                ]),
        dict(name='cmmlu-china-specific',
            subsets=[
                'cmmlu-ancient_chinese',
                'cmmlu-chinese_civil_service_exam',
                'cmmlu-chinese_driving_rule',
                'cmmlu-chinese_food_culture',
                'cmmlu-chinese_foreign_policy',
                'cmmlu-chinese_history',
                'cmmlu-chinese_literature',
                'cmmlu-chinese_teacher_qualification',
                'cmmlu-construction_project_management',
                'cmmlu-elementary_chinese',
                'cmmlu-elementary_commonsense',
                'cmmlu-ethnology',
                'cmmlu-high_school_politics',
                'cmmlu-modern_chinese',
                'cmmlu-traditional_chinese_medicine',
                ]),
        dict(name='cmmlu',
            subsets=[
                'cmmlu-agronomy',
                'cmmlu-anatomy',
                'cmmlu-ancient_chinese',
                'cmmlu-arts',
                'cmmlu-astronomy',
                'cmmlu-business_ethics',
                'cmmlu-chinese_civil_service_exam',
                'cmmlu-chinese_driving_rule',
                'cmmlu-chinese_food_culture',
                'cmmlu-chinese_foreign_policy',
                'cmmlu-chinese_history',
                'cmmlu-chinese_literature',
                'cmmlu-chinese_teacher_qualification',
                'cmmlu-college_actuarial_science',
                'cmmlu-college_education',
                'cmmlu-college_engineering_hydrology',
                'cmmlu-college_law',
                'cmmlu-college_mathematics',
                'cmmlu-college_medical_statistics',
                'cmmlu-clinical_knowledge',
                'cmmlu-college_medicine',
                'cmmlu-computer_science',
                'cmmlu-computer_security',
                'cmmlu-conceptual_physics',
                'cmmlu-construction_project_management',
                'cmmlu-economics',
                'cmmlu-education',
                'cmmlu-elementary_chinese',
                'cmmlu-elementary_commonsense',
                'cmmlu-elementary_information_and_technology',
                'cmmlu-electrical_engineering',
                'cmmlu-elementary_mathematics',
                'cmmlu-ethnology',
                'cmmlu-food_science',
                'cmmlu-genetics',
                'cmmlu-global_facts',
                'cmmlu-high_school_biology',
                'cmmlu-high_school_chemistry',
                'cmmlu-high_school_geography',
                'cmmlu-high_school_mathematics',
                'cmmlu-high_school_physics',
                'cmmlu-high_school_politics',
                'cmmlu-human_sexuality',
                'cmmlu-international_law',
                'cmmlu-journalism',
                'cmmlu-jurisprudence',
                'cmmlu-legal_and_moral_basis',
                'cmmlu-logical',
                'cmmlu-machine_learning',
                'cmmlu-management',
                'cmmlu-marketing',
                'cmmlu-marxist_theory',
                'cmmlu-modern_chinese',
                'cmmlu-nutrition',
                'cmmlu-philosophy',
                'cmmlu-professional_accounting',
                'cmmlu-professional_law',
                'cmmlu-professional_medicine',
                'cmmlu-professional_psychology',
                'cmmlu-public_relations',
                'cmmlu-security_study',
                'cmmlu-sociology',
                'cmmlu-sports_science',
                'cmmlu-traditional_chinese_medicine',
                'cmmlu-virology',
                'cmmlu-world_history',
                'cmmlu-world_religions',
                ]),
        dict(name='ceval-stem',
            subsets=[
                'ceval-computer_network',
                'ceval-operating_system',
                'ceval-computer_architecture',
                'ceval-college_programming',
                'ceval-college_physics',
                'ceval-college_chemistry',
                'ceval-advanced_mathematics',
                'ceval-probability_and_statistics',
                'ceval-discrete_mathematics',
                'ceval-electrical_engineer',
                'ceval-metrology_engineer',
                'ceval-high_school_mathematics',
                'ceval-high_school_physics',
                'ceval-high_school_chemistry',
                'ceval-high_school_biology',
                'ceval-middle_school_mathematics',
                'ceval-middle_school_biology',
                'ceval-middle_school_physics',
                'ceval-middle_school_chemistry',
                'ceval-veterinary_medicine',
                ]),
        dict(name='ceval-social-science',
            subsets=[
                'ceval-college_economics',
                'ceval-business_administration',
                'ceval-marxism',
                'ceval-mao_zedong_thought',
                'ceval-education_science',
                'ceval-teacher_qualification',
                'ceval-high_school_politics',
                'ceval-high_school_geography',
                'ceval-middle_school_politics',
                'ceval-middle_school_geography',
                ]),
        dict(name='ceval-humanities',
            subsets=[
                'ceval-modern_chinese_history',
                'ceval-ideological_and_moral_cultivation',
                'ceval-logic',
                'ceval-law',
                'ceval-chinese_language_and_literature',
                'ceval-art_studies',
                'ceval-professional_tour_guide',
                'ceval-legal_professional',
                'ceval-high_school_chinese',
                'ceval-high_school_history',
                'ceval-middle_school_history',
                ]),
        dict(name='ceval-other',
            subsets=[
                'ceval-civil_servant',
                'ceval-sports_science',
                'ceval-plant_protection',
                'ceval-basic_medicine',
                'ceval-clinical_medicine',
                'ceval-urban_and_rural_planner',
                'ceval-accountant',
                'ceval-fire_engineer',
                'ceval-environmental_impact_assessment_engineer',
                'ceval-tax_accountant',
                'ceval-physician',
                ]),
        dict(name='ceval-hard',
            subsets=[
                'ceval-advanced_mathematics',
                'ceval-discrete_mathematics',
                'ceval-probability_and_statistics',
                'ceval-college_chemistry',
                'ceval-college_physics',
                'ceval-high_school_mathematics',
                'ceval-high_school_chemistry',
                'ceval-high_school_physics',
                ]),
        dict(name='ceval',
            subsets=[
                'ceval-computer_network',
                'ceval-operating_system',
                'ceval-computer_architecture',
                'ceval-college_programming',
                'ceval-college_physics',
                'ceval-college_chemistry',
                'ceval-advanced_mathematics',
                'ceval-probability_and_statistics',
                'ceval-discrete_mathematics',
                'ceval-electrical_engineer',
                'ceval-metrology_engineer',
                'ceval-high_school_mathematics',
                'ceval-high_school_physics',
                'ceval-high_school_chemistry',
                'ceval-high_school_biology',
                'ceval-middle_school_mathematics',
                'ceval-middle_school_biology',
                'ceval-middle_school_physics',
                'ceval-middle_school_chemistry',
                'ceval-veterinary_medicine',
                'ceval-college_economics',
                'ceval-business_administration',
                'ceval-marxism',
                'ceval-mao_zedong_thought',
                'ceval-education_science',
                'ceval-teacher_qualification',
                'ceval-high_school_politics',
                'ceval-high_school_geography',
                'ceval-middle_school_politics',
                'ceval-middle_school_geography',
                'ceval-modern_chinese_history',
                'ceval-ideological_and_moral_cultivation',
                'ceval-logic',
                'ceval-law',
                'ceval-chinese_language_and_literature',
                'ceval-art_studies',
                'ceval-professional_tour_guide',
                'ceval-legal_professional',
                'ceval-high_school_chinese',
                'ceval-high_school_history',
                'ceval-middle_school_history',
                'ceval-civil_servant',
                'ceval-sports_science',
                'ceval-plant_protection',
                'ceval-basic_medicine',
                'ceval-clinical_medicine',
                'ceval-urban_and_rural_planner',
                'ceval-accountant',
                'ceval-fire_engineer',
                'ceval-environmental_impact_assessment_engineer',
                'ceval-tax_accountant',
                'ceval-physician',
                ]),
        dict(name='ceval-test-stem',
            subsets=[
                'ceval-test-computer_network',
                'ceval-test-operating_system',
                'ceval-test-computer_architecture',
                'ceval-test-college_programming',
                'ceval-test-college_physics',
                'ceval-test-college_chemistry',
                'ceval-test-advanced_mathematics',
                'ceval-test-probability_and_statistics',
                'ceval-test-discrete_mathematics',
                'ceval-test-electrical_engineer',
                'ceval-test-metrology_engineer',
                'ceval-test-high_school_mathematics',
                'ceval-test-high_school_physics',
                'ceval-test-high_school_chemistry',
                'ceval-test-high_school_biology',
                'ceval-test-middle_school_mathematics',
                'ceval-test-middle_school_biology',
                'ceval-test-middle_school_physics',
                'ceval-test-middle_school_chemistry',
                'ceval-test-veterinary_medicine',
                ]),
        dict(name='ceval-test-social-science',
            subsets=[
                'ceval-test-college_economics',
                'ceval-test-business_administration',
                'ceval-test-marxism',
                'ceval-test-mao_zedong_thought',
                'ceval-test-education_science',
                'ceval-test-teacher_qualification',
                'ceval-test-high_school_politics',
                'ceval-test-high_school_geography',
                'ceval-test-middle_school_politics',
                'ceval-test-middle_school_geography',
                ]),
        dict(name='ceval-test-humanities',
            subsets=[
                'ceval-test-modern_chinese_history',
                'ceval-test-ideological_and_moral_cultivation',
                'ceval-test-logic',
                'ceval-test-law',
                'ceval-test-chinese_language_and_literature',
                'ceval-test-art_studies',
                'ceval-test-professional_tour_guide',
                'ceval-test-legal_professional',
                'ceval-test-high_school_chinese',
                'ceval-test-high_school_history',
                'ceval-test-middle_school_history',
                ]),
        dict(name='ceval-test-other',
            subsets=[
                'ceval-test-civil_servant',
                'ceval-test-sports_science',
                'ceval-test-plant_protection',
                'ceval-test-basic_medicine',
                'ceval-test-clinical_medicine',
                'ceval-test-urban_and_rural_planner',
                'ceval-test-accountant',
                'ceval-test-fire_engineer',
                'ceval-test-environmental_impact_assessment_engineer',
                'ceval-test-tax_accountant',
                'ceval-test-physician',
                ]),
        dict(name='ceval-test-hard',
            subsets=[
                'ceval-test-advanced_mathematics',
                'ceval-test-discrete_mathematics',
                'ceval-test-probability_and_statistics',
                'ceval-test-college_chemistry',
                'ceval-test-college_physics',
                'ceval-test-high_school_mathematics',
                'ceval-test-high_school_chemistry',
                'ceval-test-high_school_physics',
                ]),
        dict(name='ceval-test',
            subsets=[
                'ceval-test-computer_network',
                'ceval-test-operating_system',
                'ceval-test-computer_architecture',
                'ceval-test-college_programming',
                'ceval-test-college_physics',
                'ceval-test-college_chemistry',
                'ceval-test-advanced_mathematics',
                'ceval-test-probability_and_statistics',
                'ceval-test-discrete_mathematics',
                'ceval-test-electrical_engineer',
                'ceval-test-metrology_engineer',
                'ceval-test-high_school_mathematics',
                'ceval-test-high_school_physics',
                'ceval-test-high_school_chemistry',
                'ceval-test-high_school_biology',
                'ceval-test-middle_school_mathematics',
                'ceval-test-middle_school_biology',
                'ceval-test-middle_school_physics',
                'ceval-test-middle_school_chemistry',
                'ceval-test-veterinary_medicine',
                'ceval-test-college_economics',
                'ceval-test-business_administration',
                'ceval-test-marxism',
                'ceval-test-mao_zedong_thought',
                'ceval-test-education_science',
                'ceval-test-teacher_qualification',
                'ceval-test-high_school_politics',
                'ceval-test-high_school_geography',
                'ceval-test-middle_school_politics',
                'ceval-test-middle_school_geography',
                'ceval-test-modern_chinese_history',
                'ceval-test-ideological_and_moral_cultivation',
                'ceval-test-logic',
                'ceval-test-law',
                'ceval-test-chinese_language_and_literature',
                'ceval-test-art_studies',
                'ceval-test-professional_tour_guide',
                'ceval-test-legal_professional',
                'ceval-test-high_school_chinese',
                'ceval-test-high_school_history',
                'ceval-test-middle_school_history',
                'ceval-test-civil_servant',
                'ceval-test-sports_science',
                'ceval-test-plant_protection',
                'ceval-test-basic_medicine',
                'ceval-test-clinical_medicine',
                'ceval-test-urban_and_rural_planner',
                'ceval-test-accountant',
                'ceval-test-fire_engineer',
                'ceval-test-environmental_impact_assessment_engineer',
                'ceval-test-tax_accountant',
                'ceval-test-physician',
                ]),
        dict(name='bbh',
            subsets=[
                'bbh-temporal_sequences',
                'bbh-disambiguation_qa',
                'bbh-date_understanding',
                'bbh-tracking_shuffled_objects_three_objects',
                'bbh-penguins_in_a_table',
                'bbh-geometric_shapes',
                'bbh-snarks',
                'bbh-ruin_names',
                'bbh-tracking_shuffled_objects_seven_objects',
                'bbh-tracking_shuffled_objects_five_objects',
                'bbh-logical_deduction_three_objects',
                'bbh-hyperbaton',
                'bbh-logical_deduction_five_objects',
                'bbh-logical_deduction_seven_objects',
                'bbh-movie_recommendation',
                'bbh-salient_translation_error_detection',
                'bbh-reasoning_about_colored_objects',
                'bbh-multistep_arithmetic_two',
                'bbh-navigate',
                'bbh-dyck_languages',
                'bbh-word_sorting',
                'bbh-sports_understanding',
                'bbh-boolean_expressions',
                'bbh-object_counting',
                'bbh-formal_fallacies',
                'bbh-causal_judgement',
                'bbh-web_of_lies',
                ]),
        dict(name='GaokaoBench',
            subsets=[
                'GaokaoBench_2010-2022_Math_II_MCQs',
                'GaokaoBench_2010-2022_Math_I_MCQs',
                'GaokaoBench_2010-2022_History_MCQs',
                'GaokaoBench_2010-2022_Biology_MCQs',
                'GaokaoBench_2010-2022_Political_Science_MCQs',
                'GaokaoBench_2010-2022_Physics_MCQs',
                'GaokaoBench_2010-2022_Chemistry_MCQs',
                'GaokaoBench_2010-2013_English_MCQs',
                'GaokaoBench_2010-2022_Chinese_Modern_Lit',
                'GaokaoBench_2010-2022_English_Fill_in_Blanks',
                'GaokaoBench_2012-2022_English_Cloze_Test',
                'GaokaoBench_2010-2022_Geography_MCQs',
                'GaokaoBench_2010-2022_English_Reading_Comp',
                'GaokaoBench_2010-2022_Chinese_Lang_and_Usage_MCQs',
                ],
            weights=dict(
                {'GaokaoBench_2010-2013_English_MCQs': 105,
                'GaokaoBench_2010-2022_Biology_MCQs': 900,
                'GaokaoBench_2010-2022_Chemistry_MCQs': 744,
                'GaokaoBench_2010-2022_Chinese_Lang_and_Usage_MCQs': 240,
                'GaokaoBench_2010-2022_Chinese_Modern_Lit': 261,
                'GaokaoBench_2010-2022_English_Fill_in_Blanks': 900.0,
                'GaokaoBench_2010-2022_English_Reading_Comp': 940,
                'GaokaoBench_2010-2022_Geography_MCQs': 380,
                'GaokaoBench_2010-2022_History_MCQs': 1148,
                'GaokaoBench_2010-2022_Math_II_MCQs': 1090,
                'GaokaoBench_2010-2022_Math_I_MCQs': 1070,
                'GaokaoBench_2010-2022_Physics_MCQs': 384,
                'GaokaoBench_2010-2022_Political_Science_MCQs': 1280,
                'GaokaoBench_2012-2022_English_Cloze_Test': 260})),
        dict(name='flores_100_Indo-European-Germanic_English',
            subsets=[
                'flores_100_afr-eng',
                'flores_100_dan-eng',
                'flores_100_deu-eng',
                'flores_100_isl-eng',
                'flores_100_ltz-eng',
                'flores_100_nld-eng',
                'flores_100_nob-eng',
                'flores_100_swe-eng',
                ]),
        dict(name='flores_100_English_Indo-European-Germanic',
            subsets=[
                'flores_100_eng-afr',
                'flores_100_eng-dan',
                'flores_100_eng-deu',
                'flores_100_eng-isl',
                'flores_100_eng-ltz',
                'flores_100_eng-nld',
                'flores_100_eng-nob',
                'flores_100_eng-swe',
                ]),
        dict(name='flores_100_Indo-European-Romance_English',
            subsets=[
                'flores_100_ast-eng',
                'flores_100_cat-eng',
                'flores_100_fra-eng',
                'flores_100_glg-eng',
                'flores_100_oci-eng',
                'flores_100_por-eng',
                'flores_100_ron-eng',
                'flores_100_spa-eng',
                ]),
        dict(name='flores_100_English_Indo-European-Romance',
            subsets=[
                'flores_100_eng-ast',
                'flores_100_eng-cat',
                'flores_100_eng-fra',
                'flores_100_eng-glg',
                'flores_100_eng-oci',
                'flores_100_eng-por',
                'flores_100_eng-ron',
                'flores_100_eng-spa',
                ]),
        dict(name='flores_100_Indo-European-Slavic_English',
            subsets=[
                'flores_100_bel-eng',
                'flores_100_bos-eng',
                'flores_100_bul-eng',
                'flores_100_ces-eng',
                'flores_100_hrv-eng',
                'flores_100_mkd-eng',
                'flores_100_pol-eng',
                'flores_100_rus-eng',
                'flores_100_slk-eng',
                'flores_100_slv-eng',
                'flores_100_srp-eng',
                'flores_100_ukr-eng',
                ]),
        dict(name='flores_100_English_Indo-European-Slavic',
            subsets=[
                'flores_100_eng-bel',
                'flores_100_eng-bos',
                'flores_100_eng-bul',
                'flores_100_eng-ces',
                'flores_100_eng-hrv',
                'flores_100_eng-mkd',
                'flores_100_eng-pol',
                'flores_100_eng-rus',
                'flores_100_eng-slk',
                'flores_100_eng-slv',
                'flores_100_eng-srp',
                'flores_100_eng-ukr',
                ]),
        dict(name='flores_100_Indo-European-Indo-Aryan_English',
            subsets=[
                'flores_100_asm-eng',
                'flores_100_ben-eng',
                'flores_100_guj-eng',
                'flores_100_hin-eng',
                'flores_100_mar-eng',
                'flores_100_npi-eng',
                'flores_100_ory-eng',
                'flores_100_pan-eng',
                'flores_100_snd-eng',
                'flores_100_urd-eng',
                ]),
        dict(name='flores_100_English_Indo-European-Indo-Aryan',
            subsets=[
                'flores_100_eng-asm',
                'flores_100_eng-ben',
                'flores_100_eng-guj',
                'flores_100_eng-hin',
                'flores_100_eng-mar',
                'flores_100_eng-npi',
                'flores_100_eng-ory',
                'flores_100_eng-pan',
                'flores_100_eng-snd',
                'flores_100_eng-urd',
                ]),
        dict(name='flores_100_Indo-European-Other_English',
            subsets=[
                'flores_100_ckb-eng',
                'flores_100_cym-eng',
                'flores_100_ell-eng',
                'flores_100_fas-eng',
                'flores_100_gle-eng',
                'flores_100_hye-eng',
                'flores_100_ita-eng',
                'flores_100_lav-eng',
                'flores_100_lit-eng',
                'flores_100_pus-eng',
                'flores_100_tgk-eng',
                ]),
        dict(name='flores_100_English_Indo-European-Other',
            subsets=[
                'flores_100_eng-ckb',
                'flores_100_eng-cym',
                'flores_100_eng-ell',
                'flores_100_eng-fas',
                'flores_100_eng-gle',
                'flores_100_eng-hye',
                'flores_100_eng-ita',
                'flores_100_eng-lav',
                'flores_100_eng-lit',
                'flores_100_eng-pus',
                'flores_100_eng-tgk',
                ]),
        dict(name='flores_100_Austronesian_English',
            subsets=[
                'flores_100_ceb-eng',
                'flores_100_ind-eng',
                'flores_100_jav-eng',
                'flores_100_mri-eng',
                'flores_100_msa-eng',
                'flores_100_tgl-eng',
                ]),
        dict(name='flores_100_English_Austronesian',
            subsets=[
                'flores_100_eng-ceb',
                'flores_100_eng-ind',
                'flores_100_eng-jav',
                'flores_100_eng-mri',
                'flores_100_eng-msa',
                'flores_100_eng-tgl',
                ]),
        dict(name='flores_100_Atlantic-Congo_English',
            subsets=[
                'flores_100_ibo-eng',
                'flores_100_kam-eng',
                'flores_100_kea-eng',
                'flores_100_lin-eng',
                'flores_100_lug-eng',
                'flores_100_nso-eng',
                'flores_100_nya-eng',
                'flores_100_sna-eng',
                'flores_100_swh-eng',
                'flores_100_umb-eng',
                'flores_100_wol-eng',
                'flores_100_xho-eng',
                'flores_100_yor-eng',
                'flores_100_zul-eng',
                ]),
        dict(name='flores_100_English_Atlantic-Congo',
            subsets=[
                'flores_100_eng-ibo',
                'flores_100_eng-kam',
                'flores_100_eng-kea',
                'flores_100_eng-lin',
                'flores_100_eng-lug',
                'flores_100_eng-nso',
                'flores_100_eng-nya',
                'flores_100_eng-sna',
                'flores_100_eng-swh',
                'flores_100_eng-umb',
                'flores_100_eng-wol',
                'flores_100_eng-xho',
                'flores_100_eng-yor',
                'flores_100_eng-zul',
                ]),
        dict(name='flores_100_Afro-Asiatic_English',
            subsets=[
                'flores_100_amh-eng',
                'flores_100_ara-eng',
                'flores_100_ful-eng',
                'flores_100_mlt-eng',
                'flores_100_orm-eng',
                'flores_100_som-eng',
                ]),
        dict(name='flores_100_English_Afro-Asiatic',
            subsets=[
                'flores_100_eng-amh',
                'flores_100_eng-ara',
                'flores_100_eng-ful',
                'flores_100_eng-mlt',
                'flores_100_eng-orm',
                'flores_100_eng-som',
                ]),
        dict(name='flores_100_Turkic_English',
            subsets=[
                'flores_100_azj-eng',
                'flores_100_kaz-eng',
                'flores_100_kir-eng',
                'flores_100_tur-eng',
                'flores_100_uzb-eng',
                ]),
        dict(name='flores_100_English_Turkic',
            subsets=[
                'flores_100_eng-azj',
                'flores_100_eng-kaz',
                'flores_100_eng-kir',
                'flores_100_eng-tur',
                'flores_100_eng-uzb',
                ]),
        dict(name='flores_100_Dravidian_English',
            subsets=[
                'flores_100_kan-eng',
                'flores_100_mal-eng',
                'flores_100_tam-eng',
                'flores_100_tel-eng',
                ]),
        dict(name='flores_100_English_Dravidian',
            subsets=[
                'flores_100_eng-kan',
                'flores_100_eng-mal',
                'flores_100_eng-tam',
                'flores_100_eng-tel',
                ]),
        dict(name='flores_100_Sino-Tibetan_English',
            subsets=[
                'flores_100_mya-eng',
                'flores_100_zho_simpl-eng',
                'flores_100_zho_trad-eng',
                ]),
        dict(name='flores_100_English_Sino-Tibetan',
            subsets=[
                'flores_100_eng-mya',
                'flores_100_eng-zho_simpl',
                'flores_100_eng-zho_trad',
                ]),
        dict(name='flores_100_Other_English',
            subsets=[
                'flores_100_est-eng',
                'flores_100_fin-eng',
                'flores_100_hau-eng',
                'flores_100_heb-eng',
                'flores_100_hun-eng',
                'flores_100_jpn-eng',
                'flores_100_kat-eng',
                'flores_100_khm-eng',
                'flores_100_kor-eng',
                'flores_100_lao-eng',
                'flores_100_luo-eng',
                'flores_100_mon-eng',
                'flores_100_tha-eng',
                'flores_100_vie-eng',
                ]),
        dict(name='flores_100_English_Other',
            subsets=[
                'flores_100_eng-est',
                'flores_100_eng-fin',
                'flores_100_eng-hau',
                'flores_100_eng-heb',
                'flores_100_eng-hun',
                'flores_100_eng-jpn',
                'flores_100_eng-kat',
                'flores_100_eng-khm',
                'flores_100_eng-kor',
                'flores_100_eng-lao',
                'flores_100_eng-luo',
                'flores_100_eng-mon',
                'flores_100_eng-tha',
                'flores_100_eng-vie',
                ]),
        dict(name='flores_100',
            subsets=[
                'flores_100_afr-eng',
                'flores_100_dan-eng',
                'flores_100_deu-eng',
                'flores_100_isl-eng',
                'flores_100_ltz-eng',
                'flores_100_nld-eng',
                'flores_100_nob-eng',
                'flores_100_swe-eng',
                'flores_100_ast-eng',
                'flores_100_cat-eng',
                'flores_100_fra-eng',
                'flores_100_glg-eng',
                'flores_100_oci-eng',
                'flores_100_por-eng',
                'flores_100_ron-eng',
                'flores_100_spa-eng',
                'flores_100_bel-eng',
                'flores_100_bos-eng',
                'flores_100_bul-eng',
                'flores_100_ces-eng',
                'flores_100_hrv-eng',
                'flores_100_mkd-eng',
                'flores_100_pol-eng',
                'flores_100_rus-eng',
                'flores_100_slk-eng',
                'flores_100_slv-eng',
                'flores_100_srp-eng',
                'flores_100_ukr-eng',
                'flores_100_asm-eng',
                'flores_100_ben-eng',
                'flores_100_guj-eng',
                'flores_100_hin-eng',
                'flores_100_mar-eng',
                'flores_100_npi-eng',
                'flores_100_ory-eng',
                'flores_100_pan-eng',
                'flores_100_snd-eng',
                'flores_100_urd-eng',
                'flores_100_ckb-eng',
                'flores_100_cym-eng',
                'flores_100_ell-eng',
                'flores_100_fas-eng',
                'flores_100_gle-eng',
                'flores_100_hye-eng',
                'flores_100_ita-eng',
                'flores_100_lav-eng',
                'flores_100_lit-eng',
                'flores_100_pus-eng',
                'flores_100_tgk-eng',
                'flores_100_ceb-eng',
                'flores_100_ind-eng',
                'flores_100_jav-eng',
                'flores_100_mri-eng',
                'flores_100_msa-eng',
                'flores_100_tgl-eng',
                'flores_100_ibo-eng',
                'flores_100_kam-eng',
                'flores_100_kea-eng',
                'flores_100_lin-eng',
                'flores_100_lug-eng',
                'flores_100_nso-eng',
                'flores_100_nya-eng',
                'flores_100_sna-eng',
                'flores_100_swh-eng',
                'flores_100_umb-eng',
                'flores_100_wol-eng',
                'flores_100_xho-eng',
                'flores_100_yor-eng',
                'flores_100_zul-eng',
                'flores_100_amh-eng',
                'flores_100_ara-eng',
                'flores_100_ful-eng',
                'flores_100_mlt-eng',
                'flores_100_orm-eng',
                'flores_100_som-eng',
                'flores_100_azj-eng',
                'flores_100_kaz-eng',
                'flores_100_kir-eng',
                'flores_100_tur-eng',
                'flores_100_uzb-eng',
                'flores_100_kan-eng',
                'flores_100_mal-eng',
                'flores_100_tam-eng',
                'flores_100_tel-eng',
                'flores_100_mya-eng',
                'flores_100_zho_simpl-eng',
                'flores_100_zho_trad-eng',
                'flores_100_est-eng',
                'flores_100_fin-eng',
                'flores_100_hau-eng',
                'flores_100_heb-eng',
                'flores_100_hun-eng',
                'flores_100_jpn-eng',
                'flores_100_kat-eng',
                'flores_100_khm-eng',
                'flores_100_kor-eng',
                'flores_100_lao-eng',
                'flores_100_luo-eng',
                'flores_100_mon-eng',
                'flores_100_tha-eng',
                'flores_100_vie-eng',
                'flores_100_eng-afr',
                'flores_100_eng-dan',
                'flores_100_eng-deu',
                'flores_100_eng-isl',
                'flores_100_eng-ltz',
                'flores_100_eng-nld',
                'flores_100_eng-nob',
                'flores_100_eng-swe',
                'flores_100_eng-ast',
                'flores_100_eng-cat',
                'flores_100_eng-fra',
                'flores_100_eng-glg',
                'flores_100_eng-oci',
                'flores_100_eng-por',
                'flores_100_eng-ron',
                'flores_100_eng-spa',
                'flores_100_eng-bel',
                'flores_100_eng-bos',
                'flores_100_eng-bul',
                'flores_100_eng-ces',
                'flores_100_eng-hrv',
                'flores_100_eng-mkd',
                'flores_100_eng-pol',
                'flores_100_eng-rus',
                'flores_100_eng-slk',
                'flores_100_eng-slv',
                'flores_100_eng-srp',
                'flores_100_eng-ukr',
                'flores_100_eng-asm',
                'flores_100_eng-ben',
                'flores_100_eng-guj',
                'flores_100_eng-hin',
                'flores_100_eng-mar',
                'flores_100_eng-npi',
                'flores_100_eng-ory',
                'flores_100_eng-pan',
                'flores_100_eng-snd',
                'flores_100_eng-urd',
                'flores_100_eng-ckb',
                'flores_100_eng-cym',
                'flores_100_eng-ell',
                'flores_100_eng-fas',
                'flores_100_eng-gle',
                'flores_100_eng-hye',
                'flores_100_eng-ita',
                'flores_100_eng-lav',
                'flores_100_eng-lit',
                'flores_100_eng-pus',
                'flores_100_eng-tgk',
                'flores_100_eng-ceb',
                'flores_100_eng-ind',
                'flores_100_eng-jav',
                'flores_100_eng-mri',
                'flores_100_eng-msa',
                'flores_100_eng-tgl',
                'flores_100_eng-ibo',
                'flores_100_eng-kam',
                'flores_100_eng-kea',
                'flores_100_eng-lin',
                'flores_100_eng-lug',
                'flores_100_eng-nso',
                'flores_100_eng-nya',
                'flores_100_eng-sna',
                'flores_100_eng-swh',
                'flores_100_eng-umb',
                'flores_100_eng-wol',
                'flores_100_eng-xho',
                'flores_100_eng-yor',
                'flores_100_eng-zul',
                'flores_100_eng-amh',
                'flores_100_eng-ara',
                'flores_100_eng-ful',
                'flores_100_eng-mlt',
                'flores_100_eng-orm',
                'flores_100_eng-som',
                'flores_100_eng-azj',
                'flores_100_eng-kaz',
                'flores_100_eng-kir',
                'flores_100_eng-tur',
                'flores_100_eng-uzb',
                'flores_100_eng-kan',
                'flores_100_eng-mal',
                'flores_100_eng-tam',
                'flores_100_eng-tel',
                'flores_100_eng-mya',
                'flores_100_eng-zho_simpl',
                'flores_100_eng-zho_trad',
                'flores_100_eng-est',
                'flores_100_eng-fin',
                'flores_100_eng-hau',
                'flores_100_eng-heb',
                'flores_100_eng-hun',
                'flores_100_eng-jpn',
                'flores_100_eng-kat',
                'flores_100_eng-khm',
                'flores_100_eng-kor',
                'flores_100_eng-lao',
                'flores_100_eng-luo',
                'flores_100_eng-mon',
                'flores_100_eng-tha',
                'flores_100_eng-vie',
                ]),
        dict(name='tydiqa-goldp',
            subsets=[
                'tydiqa-goldp_arabic',
                'tydiqa-goldp_bengali',
                'tydiqa-goldp_english',
                'tydiqa-goldp_finnish',
                'tydiqa-goldp_indonesian',
                'tydiqa-goldp_japanese',
                'tydiqa-goldp_korean',
                'tydiqa-goldp_russian',
                'tydiqa-goldp_swahili',
                'tydiqa-goldp_telugu',
                'tydiqa-goldp_thai',
                ]),
        dict(name='xiezhi',
            subsets=[
                'xiezhi-spec_eng',
                'xiezhi-spec_chn',
                'xiezhi-inter_eng',
                'xiezhi-inter_chn',
                ]),
        dict(name='scibench',
            subsets=[
                'scibench-atkins',
                'scibench-calculus',
                'scibench-chemmc',
                'scibench-class',
                'scibench-diff',
                'scibench-fund',
                'scibench-matter',
                'scibench-quan',
                'scibench-stat',
                'scibench-thermo',
                ]),
        dict(name='scibench_zs-cot',
            subsets=[
                'scibench-atkins_zs-cot',
                'scibench-calculus_zs-cot',
                'scibench-chemmc_zs-cot',
                'scibench-class_zs-cot',
                'scibench-diff_zs-cot',
                'scibench-fund_zs-cot',
                'scibench-matter_zs-cot',
                'scibench-quan_zs-cot',
                'scibench-stat_zs-cot',
                'scibench-thermo_zs-cot',
                ]),
        dict(name='scibench_fs',
            subsets=[
                'scibench-atkins_fs',
                'scibench-calculus_fs',
                'scibench-chemmc_fs',
                'scibench-class_fs',
                'scibench-diff_fs',
                'scibench-fund_fs',
                'scibench-matter_fs',
                'scibench-quan_fs',
                'scibench-stat_fs',
                'scibench-thermo_fs',
                ]),
        dict(name='scibench_fs-cot',
            subsets=[
                'scibench-atkins_fs-cot',
                'scibench-calculus_fs-cot',
                'scibench-chemmc_fs-cot',
                'scibench-class_fs-cot',
                'scibench-diff_fs-cot',
                'scibench-fund_fs-cot',
                'scibench-matter_fs-cot',
                'scibench-quan_fs-cot',
                'scibench-stat_fs-cot',
                'scibench-thermo_fs-cot',
                ]),
        dict(name='mgsm_latin',
            subsets=[
                'mgsm_de',
                'mgsm_en',
                'mgsm_es',
                'mgsm_fr',
                'mgsm_sw',
                ]),
        dict(name='mgsm_non_latin',
            subsets=[
                'mgsm_bn',
                'mgsm_ja',
                'mgsm_ru',
                'mgsm_te',
                'mgsm_th',
                'mgsm_zh',
                ]),
        dict(name='mgsm',
            subsets=[
                'mgsm_bn',
                'mgsm_de',
                'mgsm_en',
                'mgsm_es',
                'mgsm_fr',
                'mgsm_ja',
                'mgsm_ru',
                'mgsm_sw',
                'mgsm_te',
                'mgsm_th',
                'mgsm_zh',
                ]),
        dict(name='longbench_single-document-qa',
            subsets=[
                'LongBench_narrativeqa',
                'LongBench_qasper',
                'LongBench_multifieldqa_en',
                'LongBench_multifieldqa_zh',
                ]),
        dict(name='longbench_multi-document-qa',
            subsets=[
                'LongBench_hotpotqa',
                'LongBench_2wikimqa',
                'LongBench_musique',
                'LongBench_dureader',
                ]),
        dict(name='longbench_summarization',
            subsets=[
                'LongBench_gov_report',
                'LongBench_qmsum',
                'LongBench_multi_news',
                'LongBench_vcsum',
                ]),
        dict(name='longbench_few-shot-learning',
            subsets=[
                'LongBench_trec',
                'LongBench_triviaqa',
                'LongBench_samsum',
                'LongBench_lsht',
                ]),
        dict(name='longbench_synthetic-tasks',
            subsets=[
                'LongBench_passage_count',
                'LongBench_passage_retrieval_en',
                'LongBench_passage_retrieval_zh',
                ]),
        dict(name='longbench_code-completion',
            subsets=[
                'LongBench_lcc',
                'LongBench_repobench-p',
                ]),
        dict(name='longbench_zh',
            subsets=[
                'LongBench_multifieldqa_zh',
                'LongBench_dureader',
                'LongBench_vcsum',
                'LongBench_lsht',
                'LongBench_passage_retrieval_zh',
                'LongBench_lcc',
                'LongBench_repobench-p',
                ]),
        dict(name='longbench_en',
            subsets=[
                'LongBench_narrativeqa',
                'LongBench_qasper',
                'LongBench_multifieldqa_en',
                'LongBench_hotpotqa',
                'LongBench_2wikimqa',
                'LongBench_musique',
                'LongBench_gov_report',
                'LongBench_qmsum',
                'LongBench_multi_news',
                'LongBench_trec',
                'LongBench_triviaqa',
                'LongBench_samsum',
                'LongBench_passage_count',
                'LongBench_passage_retrieval_en',
                'LongBench_lcc',
                'LongBench_repobench-p',
                ]),
        dict(name='longbench',
            subsets=[
                'longbench_single-document-qa',
                'longbench_multi-document-qa',
                'longbench_summarization',
                'longbench_few-shot-learning',
                'longbench_synthetic-tasks',
                'longbench_code-completion',
                ]),
        ])
work_dir='./base_1.5B_eval/20241231_200238'