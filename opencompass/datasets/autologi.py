# flake8: noqa
# yapf: disable
"""OpenCompass support for the AutoLogi logic-puzzle benchmark.

AutoLogi (arXiv:2502.16906) grades open-ended logic puzzles by program
verification: every puzzle ships Python source strings that validate the
answer shape and encode the logical constraints, and a model answer is
correct only when it satisfies all of them. The verifier here is a
clean-room reimplementation from the paper and the public data schema.

Execution safety: the only code ever executed is the dataset-packaged
check/constraint source from the Apache-2.0 ``qzhu/AutoLogi`` data, and
it runs in a forked subprocess with a timeout (mirroring
``opencompass/datasets/cruxeval.py``). The model generation itself is
treated strictly as data — it is parsed with ``ast.literal_eval`` and
embedded back via ``repr``, never executed.
"""
import ast
import io
import json
import multiprocessing
import os
import re
from contextlib import redirect_stdout

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

HF_REPO_ID = 'qzhu/AutoLogi'
DATASET_ID = 'opencompass/autologi'
LANG_TO_FILE = {
    'en': 'AutoLogi_en.jsonl',
    'cn': 'AutoLogi_cn.jsonl',
}
KEEP_COLUMNS = ('prompt', 'code', 'id')

_FENCED_BLOCK_RE = re.compile(r'```([A-Za-z0-9_+\-.]*)[ \t]*\n?(.*?)```',
                              re.DOTALL)
_HUB_ID_RE = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')
_JSON_LIKE_RE = re.compile(r'\b(true|false|null)\b')
_JSON_LIKE_MAP = {'true': 'True', 'false': 'False', 'null': 'None'}

_VERIFY_FUNCTION_DEF = '''

def verify_function(answer, inputs_check_fn, constraint_fns):
    """Return True when ``answer`` passes every puzzle check."""
    if not inputs_check_fn(answer):
        return False
    for constraint_fn in constraint_fns:
        if not constraint_fn(answer):
            return False
    return True
'''


def _hub_file_url(repo_id: str, filename: str) -> str:
    return f'https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}'


def _resolve_data_file(path: str, filename: str) -> str:
    """Resolve ``filename`` for the dataset id or path ``path``.

    Prefers a file already present in a cache directory resolved through
    ``get_data_path``; otherwise the data is fetched at runtime from the
    Apache-2.0 Hugging Face repository (the benchmark is not
    redistributed with OpenCompass).
    """
    if path.startswith(('http://', 'https://')) or path.endswith('.jsonl'):
        return path
    try:
        resolved = get_data_path(path)
    except Exception:
        resolved = ''
    if resolved and os.path.isfile(resolved):
        return resolved
    if resolved and os.path.isdir(resolved):
        candidate = os.path.join(resolved, filename)
        if os.path.isfile(candidate):
            return candidate
    if resolved and _HUB_ID_RE.fullmatch(resolved):
        # get_data_path returned a hub repository id (DATASET_SOURCE=HF)
        return _hub_file_url(resolved, filename)
    # unmapped id or missing local cache: load from the HF repository.
    # An explicit hub id (e.g. 'qzhu/AutoLogi') is honored; the internal
    # mapping id resolves to the registered hub dataset.
    if path == DATASET_ID or not _HUB_ID_RE.fullmatch(path):
        repo_id = HF_REPO_ID
    else:
        repo_id = path
    return _hub_file_url(repo_id, filename)


def _last_balanced(text: str, open_ch: str, close_ch: str):
    """Return the last balanced ``open_ch ... close_ch`` span in ``text``.

    Returns None when no balanced span exists. Brackets inside quoted
    strings may confuse the counting; such candidates simply fail to
    parse and count as extraction failures.
    """
    end = text.rfind(close_ch)
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            char = text[start]
            if char == close_ch:
                depth += 1
            elif char == open_ch:
                depth -= 1
                if depth == 0:
                    return text[start:end + 1]
        end = text.rfind(close_ch, 0, end)
    return None


def _parse_literal(candidate: str):
    """Parse ``candidate`` as a Python literal, or return None."""
    sources = [candidate]
    if _JSON_LIKE_RE.search(candidate):
        # tolerate JSON-style booleans / null on a retry
        sources.append(
            _JSON_LIKE_RE.sub(
                lambda match: _JSON_LIKE_MAP[match.group(1)], candidate))
    for source in sources:
        try:
            return ast.literal_eval(source)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
    return None


def autologi_postprocess(text: str):
    """Extract the answer literal from a model generation.

    Follows the official grading protocol: take the last fenced code
    block when present, else the last balanced ``{...}``, else the last
    balanced ``[...]``, and parse it with ``ast.literal_eval``. The
    generation is never executed. Returns None when no answer can be
    extracted; such predictions simply count as incorrect.
    """
    if not isinstance(text, str):
        text = str(text)
    blocks = _FENCED_BLOCK_RE.findall(text)
    scope = blocks[-1][1] if blocks else text
    candidates = [scope.strip()]
    for open_ch, close_ch in (('{', '}'), ('[', ']')):
        span = _last_balanced(scope, open_ch, close_ch)
        if span is not None:
            candidates.append(span)
    for candidate in candidates:
        answer = _parse_literal(candidate)
        if answer is not None:
            return answer
    return None


def build_verifier_program(code: dict, answer) -> str:
    """Assemble the verification program for one puzzle.

    Layout: the dataset-packaged ``inputs_check`` source, the constraint
    predicates (ending in a module-level ``constraint_list`` binding),
    the shared ``verify_function`` wrapper, then the invocation printing
    the result. ``answer`` is embedded via ``repr`` so it stays data.
    """
    return (f"{code['Inputs_Check_code']}\n"
            f"{code['Constraint_List_code']}\n"
            f'{_VERIFY_FUNCTION_DEF}\n'
            f'res = verify_function({answer!r}, inputs_check, '
            'constraint_list)\n'
            'print(res)\n')


def _autologi_exec(program: str, queue) -> None:
    """Execute the verifier program in a subprocess to allow timeout."""
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(program, {'__builtins__': __builtins__})
        queue.put(buffer.getvalue())
    except Exception:
        queue.put('')


def check_answer(code: dict, answer, timeout: int = 10) -> bool:
    """Verify ``answer`` against the puzzle's packaged checks."""
    if not isinstance(code, dict) or 'Inputs_Check_code' not in code \
            or 'Constraint_List_code' not in code:
        return False
    program = build_verifier_program(code, answer)
    try:
        try:
            ctx = multiprocessing.get_context('fork')
        except ValueError:
            ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        proc = ctx.Process(target=_autologi_exec, args=(program, queue))
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False
        return queue.get_nowait().strip() == 'True'
    except Exception:
        return False


@LOAD_DATASET.register_module()
class AutoLogiDataset(BaseDataset):

    @staticmethod
    def load(path: str = DATASET_ID, lang: str = 'en'):
        """Load the AutoLogi puzzles for ``lang`` (``'en'`` or ``'cn'``).

        The language split is chosen by FILENAME (``AutoLogi_en.jsonl``
        vs ``AutoLogi_cn.jsonl``): the per-record ``lang`` field is
        mislabeled ``"zh_cn"`` even in the English file, so it cannot be
        trusted for splitting.
        """
        from datasets import load_dataset

        if lang not in LANG_TO_FILE:
            raise ValueError(
                f"lang must be one of {sorted(LANG_TO_FILE)}, got '{lang}'")
        dataset = load_dataset(
            'json',
            data_files={'test': _resolve_data_file(path, LANG_TO_FILE[lang])},
            split='test')
        keep = [c for c in KEEP_COLUMNS if c in dataset.column_names]
        return dataset.remove_columns(
            [c for c in dataset.column_names if c not in keep])


@ICL_EVALUATORS.register_module()
class AutoLogiEvaluator(BaseEvaluator):
    """Program-verified evaluator for AutoLogi (pass@1 accuracy).

    For each sample the extracted answer is checked by running the
    puzzle's packaged ``inputs_check`` and every constraint predicate in
    a forked subprocess with a timeout. A prediction passes only when
    the verification program prints ``True``; extraction failures and
    timeouts count as incorrect.
    """

    def __init__(self, timeout: int = 10) -> None:
        super().__init__()
        self.timeout = timeout

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        details = {}
        correct = 0
        for idx, prediction in enumerate(predictions):
            item = test_set[idx]
            code = item['code']
            if isinstance(code, str):
                code = json.loads(code)
            answer = autologi_postprocess(prediction)
            is_correct = (answer is not None
                          and check_answer(code, answer, self.timeout))
            correct += int(is_correct)

            details[str(idx)] = {
                'id': item.get('id', idx),
                'prediction': prediction,
                'is_correct': is_correct,
            }

        total = len(predictions)
        accuracy = correct / total * 100 if total > 0 else 0.0
        return {'accuracy': accuracy, 'details': details}
