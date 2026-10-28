"""七任务适配与监督边界；CPU 验证，文献数值不作为本地运行结果。"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from test_phi2_contract import tiny_model
from test_phi2_training import tokenizer
from phi2_benchmark_tasks import training_pair,load_protocol
from prepare_phi2_benchmark_data import encode_pair


class BenchmarkTests(unittest.TestCase):
    def test_answer_labels_are_response_only_and_overlength_is_filtered(self):
        tok=tokenizer()
        ids,labels=encode_pair('w1 w2',' w3 w4',tok,32)
        self.assertEqual(labels,[-100,-100,*ids[2:]])
        self.assertEqual(ids[-1],tok.eos_token_id)
        self.assertIsNone(encode_pair(' '.join(['w1']*40),' w3',tok,32))

    def test_question_dedup_fingerprint_does_not_depend_on_eval_answer(self):
        standard=SimpleNamespace(doc_to_text=lambda d:'question',doc_to_target=lambda d:d['gold'],
             doc_to_choice=lambda d:['no','yes'],config=SimpleNamespace(target_delimiter=' '))
        self.assertEqual(training_pair(standard,{'gold':0})[3],training_pair(standard,{'gold':1})[3])
        winogrande=SimpleNamespace(doc_to_text=lambda d:d['gold'],doc_to_target=lambda d:'shared suffix',
             doc_to_choice=lambda d:['prefix A','prefix B'],config=SimpleNamespace(target_delimiter=' '))
        self.assertEqual(training_pair(winogrande,{'gold':0})[3],training_pair(winogrande,{'gold':1})[3])
        self.assertEqual(training_pair(winogrande,{'gold':1})[:2],('prefix B',' shared suffix'))

    def test_protocol_contains_seven_and_paper_numbers_are_external(self):
        self.assertEqual(len(load_protocol()),7)
        paper=json.loads((Path(__file__).parent/'paper_phi2_reference.json').read_text())
        self.assertAlmostEqual(sum(paper['dense_phi2_percent'])/7,71.09)
        self.assertIn('not_local',paper['kind'])

    @unittest.skipUnless(os.environ.get('PHI2_EVAL_PYTHON'),'需要隔离的官方 harness 环境')
    def test_official_harness_runs_all_seven_on_tiny_phi_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'tiny';base.mkdir()
            tiny_model().save_pretrained(base);tokenizer().save_pretrained(base)
            output=root/'evaluation'
            script=Path(__file__).parent/'evaluate_phi2_benchmarks.py'
            command=[os.environ['PHI2_EVAL_PYTHON'],str(script),'--model-kind','dense','--model-dir',str(base),
                     '--output',str(output),'--device','cpu','--limit','2']
            result=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=180)
            self.assertEqual(result.returncode,0,result.stdout[-1200:]+'\n'+result.stderr[-3000:])
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['tasks']),7);self.assertTrue(summary['functional_only'])
            self.assertFalse(summary['full_seven_task_evaluation'])
            for name in summary['tasks']:
                report=json.loads((output/(name+'.json')).read_text())
                self.assertEqual(report['rows'],2)
                self.assertEqual(len((output/(name+'_samples.jsonl')).read_text().splitlines()),2)
            before=(output/'summary.json').read_bytes()
            before_names={p.name for p in output.iterdir()}
            second=subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(second.returncode,0);self.assertEqual((output/'summary.json').read_bytes(),before)
            self.assertEqual({p.name for p in output.iterdir()},before_names)


if __name__=='__main__':unittest.main()
