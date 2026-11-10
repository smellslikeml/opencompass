"""AutoLogi: program-verified logic-puzzle reasoning benchmark."""

from opencompass.datasets import AutoLogiDataset, AutoLogiEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

DATASET_PATH = 'opencompass/autologi'

autologi_datasets = []
for lang in ('en', 'cn'):
    autologi_reader_cfg = dict(
        input_columns=['prompt'],
        output_column='code',
    )

    autologi_infer_cfg = dict(
        prompt_template=dict(
            type=RawPromptTemplate,
            # each record's prompt is the complete task instruction and is
            # passed through verbatim, without any wrapper transformation
            messages=[dict(role='user', content='{prompt}')],
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=2048),
    )

    autologi_eval_cfg = dict(
        evaluator=dict(type=AutoLogiEvaluator),
    )

    autologi_datasets.append(
        dict(
            type=AutoLogiDataset,
            abbr=f'autologi-{lang}',
            path=DATASET_PATH,
            lang=lang,
            reader_cfg=autologi_reader_cfg,
            infer_cfg=autologi_infer_cfg,
            eval_cfg=autologi_eval_cfg,
        ))
