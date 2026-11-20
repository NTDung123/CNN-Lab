"""Operational regressions: malformed reports, safe scopes and model recovery."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app import app
from vision.report_files import write_json_atomic,evaluation_directory,annotate_confidence_policy
from vision.models import SmallCNN
from scripts.train_geometry import validation_detector
from scripts.evaluate_real import evaluate_real_report
import scripts.compare_segmentation as comparison
import scripts.evaluate as evaluation


def report_fixture():
    score={'tp':1,'fp':0,'fn':0,'confidence_threshold':.75}
    return {'asset_identity':{},'audit':{},'protocol':{},'detection_all':copy.deepcopy(score),
            'detection_by_role':{'development':copy.deepcopy(score)},'detection_by_difficulty':{},
            'cases':[{'id':'a','split':'development','segmentation':{'background_border_std':20.},
                      'detection':copy.deepcopy(score),'comparison_image':'frozen.png'}]}


class OperationalAuditTests(unittest.TestCase):
    def test_valid_json_with_wrong_structure_returns_json_503(self):
        app.config['TESTING']=True
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir()
            for route,name in [('/api/metrics','evaluation.json'),('/api/real','real_evaluation.json')]:
                for value in ([],None,'text',42,{'asset_identity':[]},{'digits':[],'shapes':{}}):
                    with self.subTest(route=route,value=value):
                        (root/'results'/name).write_text(json.dumps(value))
                        with patch('app.ROOT',root):response=app.test_client().get(route)
                        self.assertEqual(response.status_code,503);self.assertIn('error',response.get_json())

    def test_development_report_preserves_full_test_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();main=root/'results/real_evaluation.json';main.write_text('original complete report')
            fake=SimpleNamespace(profile={'min_confidence':.75,'complex_min_confidence':.9})
            with patch('scripts.evaluate_real.evaluate_real_cases',return_value=report_fixture()) as core:
                result=evaluate_real_report(fake,root)
            self.assertFalse(core.call_args.kwargs['write']);self.assertEqual(main.read_text(),'original complete report')
            saved=json.loads((root/'results/real_development_evaluation.json').read_text())
            self.assertEqual(saved['report_scope'],'development_validation');self.assertIsNone(saved['cases'][0]['comparison_image'])

    def test_exploratory_report_preserves_full_test_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();main=root/'results/real_evaluation.json';main.write_text('frozen')
            fake=SimpleNamespace(profile={'min_confidence':.75,'complex_min_confidence':.9})
            with patch('scripts.evaluate_real.evaluate_real_cases',return_value=report_fixture()) as core:
                result=evaluate_real_report(fake,root,exploratory=True,include_test=True)
            self.assertFalse(core.call_args.kwargs['write']);self.assertEqual(main.read_text(),'frozen')
            self.assertTrue((root/'results/real_exploratory_evaluation.json').exists())
            self.assertEqual(result['report_scope'],'exploratory')

    def test_explicit_full_evaluation_publishes_complete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            fake=SimpleNamespace(profile={'min_confidence':.75,'complex_min_confidence':.9})
            with patch('scripts.evaluate_real.evaluate_real_cases',return_value=report_fixture()) as core:
                result=evaluate_real_report(fake,Path(folder),include_test=True)
            self.assertTrue(core.call_args.kwargs['write']);self.assertTrue((Path(folder)/'results/real_evaluation.json').exists())
            self.assertEqual(result['report_scope'],'frozen_complete_test')
            self.assertEqual(result['cases'][0]['comparison_image'],'frozen.png')

    def test_development_comparison_cannot_replace_complete_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();main=root/'results/segmentation_comparison.json';main.write_text('frozen comparison')
            fake=SimpleNamespace(shapes=None,shape_metadata={},shape_classes=['circle','rectangle','triangle','ellipse'],profile={'min_confidence':.75})
            with patch.object(comparison,'ROOT',root),patch.object(comparison,'Recognizer',return_value=fake),patch.object(comparison,'legacy_recognizer',return_value=fake),patch.object(comparison,'phase4_recognizer',return_value=fake),patch.object(comparison,'read_real_manifest',return_value={'cases':[]}),patch.object(comparison,'asset_identity',return_value={}),patch.object(comparison,'file_sha256',return_value='fixture'):
                comparison.compare(False)
            self.assertEqual(main.read_text(),'frozen comparison');self.assertTrue((root/'results/segmentation_comparison_development.json').exists())

    def test_report_records_actual_adaptive_confidence_without_changing_scores(self):
        result=report_fixture();before=result['detection_all'].copy()
        annotate_confidence_policy(result,{'min_confidence':.75,'complex_min_confidence':.9,'illumination_border_std':12})
        self.assertEqual(result['cases'][0]['detection']['confidence_threshold'],.9)
        self.assertIsNone(result['detection_all']['confidence_threshold'])
        self.assertEqual(result['protocol']['confidence_policy']['base_threshold'],.75)
        for key in ('tp','fp','fn'):self.assertEqual(result['detection_all'][key],before[key])

    def test_nonfinite_report_does_not_destroy_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';path.write_text('previous')
            with self.assertRaises(ValueError):write_json_atomic(path,{'value':float('nan')})
            self.assertEqual(path.read_text(),'previous')

    def test_failed_atomic_replace_preserves_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';path.write_text('previous')
            with patch('vision.report_files.os.replace',side_effect=OSError('locked')):
                with self.assertRaises(OSError):write_json_atomic(path,{'valid':1})
            self.assertEqual(path.read_text(),'previous');self.assertEqual(list(Path(folder).glob('*.tmp')),[])

    def test_training_validation_detector_does_not_need_deployed_weights(self):
        with patch('vision.service.load_cnn',side_effect=AssertionError('must not load a deployed checkpoint')):
            detector=validation_detector(SmallCNN(5,32))
            image=np.tile(np.arange(100,dtype=np.uint8)*2+30,(80,1))
            result=detector.geometry(image,previews=False)
        self.assertEqual(result['min_confidence'],.75)
        self.assertEqual(detector.profile['complex_min_confidence'],.75)

    def test_geometry_training_recovers_when_weights_directory_is_missing(self):
        import scripts.train_geometry as trainer
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data').mkdir()
            images=np.zeros((4,32,32),np.float32);images[:,8:24,8:24]=1
            labels=np.array([0,1,2,3],np.int64)
            np.savez(root/'data/real_training.npz',x_train=images,y_train=labels,x_real_validation=images,y_real_validation=labels)
            geometry={'archive_sha256':'fixture','protocol_version':2,'splits':{'train':{'split_signature':'a'},'validation':{'split_signature':'b'}}}
            (root/'data/geometry_splits.json').write_text(json.dumps(geometry))
            extra={'archive_sha256':'fixture','dependencies':{'manifest_sha256':'fixture'},'real_validation_class_counts':{'0':1,'1':1,'2':1,'3':1}}
            args=SimpleNamespace(seed=2,threads=1,lr=.0001,epochs=1,batch_size=4)
            with patch.object(trainer,'ROOT',root),patch.object(trainer,'prepare_real_training',return_value=extra),patch.object(trainer,'load_shape_split',return_value=(images,labels)),patch.object(trainer,'read_real_manifest',return_value={'cases':[]}):
                trainer.train_geometry(args)
            self.assertTrue((root/'weights/shapes_cnn.pt').is_file())
            self.assertTrue((root/'results/training_shapes.json').is_file())

    def test_exploratory_failure_images_do_not_replace_main_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();main=root/'results/failures.png';main.write_bytes(b'frozen image')
            scoped=evaluation_directory(root,exploratory=True);scoped.mkdir()
            with patch.object(evaluation,'RESULTS',scoped):
                evaluation.failure_grid(np.zeros((2,28,28),np.float32),np.array([0,1]),np.array([[0.,1.],[0.,1.]]),'failures.png')
            self.assertEqual(main.read_bytes(),b'frozen image');self.assertTrue((scoped/'failures.png').exists())

    def test_unsupported_geometry_smoke_flags_fail_before_training(self):
        import hashlib
        checkpoint=ROOT/'weights/shapes_cnn.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for args in (['--limit-train','4'],['--shape-samples','40'],['--lr','nan']):
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--task','shapes',*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),before)


if __name__=='__main__':unittest.main(verbosity=2)
