import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from opencompass.datasets import AutoLogiDataset, AutoLogiEvaluator
from opencompass.datasets.autologi import (_resolve_data_file,
                                           autologi_postprocess,
                                           build_verifier_program,
                                           check_answer)
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.datasets_info import DATASETS_MAPPING

# Fixtures mirror the qzhu/AutoLogi schema: each record carries the task
# `prompt`, a `code` dict of Python source strings (`Inputs_Check_code`,
# `Constraint_List_code`, plus the unused `Traverse_code`), an `example`,
# and an `id`. Per the paper, `example` demonstrates the answer FORMAT
# ("an example of valid input, not necessarily the correct arrangement"),
# so a format-valid answer that violates a constraint must be judged
# incorrect.

ORDER_CODE = {
    'Inputs_Check_code': ('def inputs_check(inputs):\n'
                          '    if not isinstance(inputs, list):\n'
                          '        return False\n'
                          '    if len(inputs) != 3:\n'
                          '        return False\n'
                          '    if len(set(inputs)) != len(inputs):\n'
                          '        return False\n'
                          '    return True\n'),
    'Constraint_List_code':
    ('def constraint_apple_before_banana(inputs):\n'
     '    if inputs.index("banana") < inputs.index("apple"):\n'
     '        return False\n'
     '    return True\n'
     '\n'
     'constraint_list = [constraint_apple_before_banana]\n'),
    'Traverse_code':
    'def traverse(inputs):\n    return True\n',
}
ORDER_EXAMPLE = ['apple', 'banana', 'cherry']
ORDER_VIOLATION = ['banana', 'apple', 'cherry']

SEAT_CODE = {
    'Inputs_Check_code':
    ('def inputs_check(inputs):\n'
     '    if not isinstance(inputs, dict):\n'
     '        return False\n'
     '    if set(inputs.keys()) != {"driver", "navigator"}:\n'
     '        return False\n'
     '    if inputs["driver"] not in ("amy", "ben", "cal"):\n'
     '        return False\n'
     '    if inputs["navigator"] not in ("amy", "ben", "cal"):\n'
     '        return False\n'
     '    if inputs["driver"] == inputs["navigator"]:\n'
     '        return False\n'
     '    return True\n'),
    'Constraint_List_code': ('def constraint_driver_is_amy(inputs):\n'
                             '    if inputs["driver"] != "amy":\n'
                             '        return False\n'
                             '    return True\n'
                             '\n'
                             'def constraint_navigator_not_cal(inputs):\n'
                             '    if inputs["navigator"] == "cal":\n'
                             '        return False\n'
                             '    return True\n'
                             '\n'
                             'constraint_list = [constraint_driver_is_amy, '
                             'constraint_navigator_not_cal]\n'),
    'Traverse_code':
    'def traverse(inputs):\n    return True\n',
}
SEAT_EXAMPLE = {'driver': 'amy', 'navigator': 'ben'}
SEAT_VIOLATION = {'driver': 'ben', 'navigator': 'cal'}

LOOP_CODE = {
    'Inputs_Check_code': ('def inputs_check(inputs):\n'
                          '    if not isinstance(inputs, list):\n'
                          '        return False\n'
                          '    return True\n'),
    'Constraint_List_code': ('def constraint_forever(inputs):\n'
                             '    while True:\n'
                             '        pass\n'
                             '    return True\n'
                             '\n'
                             'constraint_list = [constraint_forever]\n'),
}


def _record(idx, code, example):
    # `lang` is deliberately mislabeled "zh_cn" even in the English file,
    # as in the real data — the split must follow the FILENAME instead.
    # `example` is kept as a JSON string so heterogeneous solution types
    # (list vs dict) stay column-compatible.
    return dict(
        id=idx,
        lang='zh_cn',
        prompt=f'Puzzle {idx}: give a valid assignment.',
        code=code,
        example=json.dumps(example),
    )


class AutoLogiExtractionTests(unittest.TestCase):

    def test_last_fenced_block_wins(self):
        text = ('First guess:\n```python\n' + repr(ORDER_VIOLATION) +
                '\n```\nWait, correcting:\n```json\n'
                '["apple", "banana", "cherry"]\n```')
        self.assertEqual(autologi_postprocess(text), ORDER_EXAMPLE)

    def test_prose_with_unfenced_list(self):
        text = ('The order should be ' + repr(ORDER_EXAMPLE) +
                ' as explained above.')
        self.assertEqual(autologi_postprocess(text), ORDER_EXAMPLE)

    def test_balanced_dict_precedes_list(self):
        text = 'result: [1, 2] but the mapping is {"a": 1}'
        self.assertEqual(autologi_postprocess(text), {'a': 1})

    def test_json_style_booleans_and_null(self):
        text = '```json\n{"driver": "amy", "ok": true, "note": null}\n```'
        self.assertEqual(autologi_postprocess(text), {
            'driver': 'amy',
            'ok': True,
            'note': None
        })

    def test_garbage_returns_none_without_crashing(self):
        self.assertIsNone(autologi_postprocess('I cannot solve this puzzle.'))
        self.assertIsNone(autologi_postprocess(''))
        self.assertIsNone(autologi_postprocess('answer: ```unfinished'))

    def test_verifier_program_layout(self):
        program = build_verifier_program(ORDER_CODE, ORDER_EXAMPLE)
        self.assertIn(ORDER_CODE['Inputs_Check_code'], program)
        self.assertIn(ORDER_CODE['Constraint_List_code'], program)
        self.assertIn('def verify_function(', program)
        self.assertIn(
            f'verify_function({ORDER_EXAMPLE!r}, inputs_check, '
            'constraint_list)', program)


class AutoLogiVerifierTests(unittest.TestCase):

    def test_record_example_passes(self):
        self.assertTrue(check_answer(ORDER_CODE, ORDER_EXAMPLE, timeout=5))
        self.assertTrue(check_answer(SEAT_CODE, SEAT_EXAMPLE, timeout=5))

    def test_constraint_violation_fails(self):
        self.assertFalse(check_answer(ORDER_CODE, ORDER_VIOLATION, timeout=5))
        self.assertFalse(check_answer(SEAT_CODE, SEAT_VIOLATION, timeout=5))

    def test_malformed_answer_fails_inputs_check(self):
        self.assertFalse(
            check_answer(ORDER_CODE, ['apple', 'banana'], timeout=5))

    def test_infinite_loop_times_out(self):
        start = time.monotonic()
        self.assertFalse(check_answer(LOOP_CODE, ['x'], timeout=1))
        self.assertLess(time.monotonic() - start, 30)

    def test_missing_verifier_source_is_incorrect(self):
        self.assertFalse(
            check_answer({'Inputs_Check_code': 'x = 1'},
                         ORDER_EXAMPLE,
                         timeout=5))

    def test_format_example_violating_constraint_is_incorrect(self):
        # Mirrors a real qzhu/AutoLogi record: the bundled `example` is a
        # FORMAT exemplar, and here it violates the only bound constraint
        # (灏韵站 must come immediately before 韮上站).
        code = {
            'Inputs_Check_code':
            ('def inputs_check(inputs):\n'
             '    if not isinstance(inputs, list):\n'
             '        return False\n'
             '    if len(inputs) != 5:\n'
             '        return False\n'
             '    if set(inputs) != {"灏韵站", "胡瑶站", "扶夷站", '
             '"韮上站", "银岭站"}:\n'
             '        return False\n'
             '    return True\n'),
            'Constraint_List_code': ('def constraint_3(inputs):\n'
                                     '    jiu_index = inputs.index("韮上站")\n'
                                     '    hao_index = inputs.index("灏韵站")\n'
                                     '    if not (hao_index < jiu_index and '
                                     'abs(jiu_index - hao_index) == 1):\n'
                                     '        return False\n'
                                     '    return True\n'
                                     '\n'
                                     'constraint_list = [constraint_3]\n'),
        }
        format_example = ['银岭站', '韮上站', '灏韵站', '扶夷站', '胡瑶站']
        solution = ['银岭站', '灏韵站', '韮上站', '扶夷站', '胡瑶站']
        self.assertFalse(check_answer(code, format_example, timeout=5))
        self.assertTrue(check_answer(code, solution, timeout=5))


class AutoLogiEvaluatorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.records = [
            _record(1, ORDER_CODE, ORDER_EXAMPLE),
            _record(2, SEAT_CODE, SEAT_EXAMPLE),
        ]
        cls.test_set = Dataset.from_list(cls.records)
        cls.evaluator = AutoLogiEvaluator(timeout=5)

    def test_end_to_end_half_correct(self):
        predictions = [
            # correct: prose + fenced example for record 1
            'Reasoning...\n```python\n' + repr(ORDER_EXAMPLE) + '\n```',
            # incorrect: no extractable answer for record 2
            'Sorry, I do not know.',
        ]
        result = self.evaluator.score(predictions, ['code', 'code'],
                                      self.test_set)
        self.assertEqual(result['accuracy'], 50.0)
        details = result['details']
        self.assertEqual(sorted(details), ['0', '1'])
        self.assertTrue(details['0']['is_correct'])
        self.assertFalse(details['1']['is_correct'])
        self.assertEqual(details['0']['id'], 1)
        self.assertEqual(details['1']['id'], 2)

    def test_all_correct_and_all_wrong(self):
        predictions = [
            '```python\n' + repr(ORDER_EXAMPLE) + '\n```',
            'The seats: ```json\n{"driver": "amy", "navigator": "ben"}\n```',
        ]
        result = self.evaluator.score(predictions, ['code', 'code'],
                                      self.test_set)
        self.assertEqual(result['accuracy'], 100.0)

        result = self.evaluator.score(predictions[::-1], ['code', 'code'],
                                      self.test_set)
        self.assertEqual(result['accuracy'], 0.0)

    def test_length_mismatch_reports_error(self):
        result = self.evaluator.score(['x'], ['code', 'code'], self.test_set)
        self.assertIn('error', result)

    def test_code_column_may_be_json_string(self):
        string_rows = [
            dict(record, code=json.dumps(record['code']))
            for record in self.records
        ]
        predictions = [
            '```python\n' + repr(ORDER_EXAMPLE) + '\n```',
            'no answer',
        ]
        result = self.evaluator.score(predictions, ['code', 'code'],
                                      Dataset.from_list(string_rows))
        self.assertEqual(result['accuracy'], 50.0)


class AutoLogiDatasetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls.temp_dir.name)
        en_rows = [
            _record(1, ORDER_CODE, ORDER_EXAMPLE),
            _record(2, SEAT_CODE, SEAT_EXAMPLE),
        ]
        cn_rows = [_record(3, ORDER_CODE, ORDER_EXAMPLE)]
        (cls.data_dir / 'AutoLogi_en.jsonl').write_text(''.join(
            json.dumps(row) + '\n' for row in en_rows),
                                                        encoding='utf-8')
        (cls.data_dir / 'AutoLogi_cn.jsonl').write_text(''.join(
            json.dumps(row) + '\n' for row in cn_rows),
                                                        encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_load_splits_by_filename_not_lang_field(self):
        with patch('opencompass.datasets.autologi.get_data_path',
                   return_value=str(self.data_dir)):
            en = AutoLogiDataset.load(path='opencompass/autologi', lang='en')
            cn = AutoLogiDataset.load(path='opencompass/autologi', lang='cn')
        self.assertEqual(len(en), 2)
        self.assertEqual(len(cn), 1)
        self.assertEqual(set(en.column_names), {'prompt', 'code', 'id'})
        self.assertIsInstance(en[0]['code'], dict)
        self.assertIn('Inputs_Check_code', en[0]['code'])

    def test_load_rejects_unknown_lang(self):
        with self.assertRaisesRegex(ValueError, 'lang'):
            AutoLogiDataset.load(path=str(self.data_dir), lang='fr')

    def test_resolve_prefers_local_file(self):
        with patch('opencompass.datasets.autologi.get_data_path',
                   return_value=str(self.data_dir)):
            resolved = _resolve_data_file('opencompass/autologi',
                                          'AutoLogi_cn.jsonl')
        self.assertEqual(resolved, str(self.data_dir / 'AutoLogi_cn.jsonl'))

    def test_resolve_falls_back_to_hf_repository(self):
        with patch('opencompass.datasets.autologi.get_data_path',
                   side_effect=AssertionError('no local data')):
            resolved = _resolve_data_file('opencompass/autologi',
                                          'AutoLogi_en.jsonl')
        self.assertEqual(
            resolved,
            'https://huggingface.co/datasets/qzhu/AutoLogi/resolve/main/'
            'AutoLogi_en.jsonl')

    def test_resolve_hub_repo_id_from_mapping(self):
        with patch('opencompass.datasets.autologi.get_data_path',
                   return_value='qzhu/AutoLogi'):
            resolved = _resolve_data_file('opencompass/autologi',
                                          'AutoLogi_en.jsonl')
        self.assertEqual(
            resolved,
            'https://huggingface.co/datasets/qzhu/AutoLogi/resolve/main/'
            'AutoLogi_en.jsonl')


class AutoLogiRegistrationTests(unittest.TestCase):

    def test_modules_are_registered(self):
        self.assertIs(LOAD_DATASET.get('AutoLogiDataset'), AutoLogiDataset)
        self.assertIs(ICL_EVALUATORS.get('AutoLogiEvaluator'),
                      AutoLogiEvaluator)

    def test_datasets_mapping_entry(self):
        entry = DATASETS_MAPPING['opencompass/autologi']
        self.assertEqual(entry['hf_id'], 'qzhu/AutoLogi')
        self.assertIsNone(entry['ms_id'])
        self.assertEqual(entry['local'], './data/autologi')


if __name__ == '__main__':
    unittest.main()
