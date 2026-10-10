import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import cv2
import numpy as np
from import_phone_recordings import import_recordings
from summarize_ball_hypotheses import summarize
import test_summarize_ball_hypotheses as summary_test_fixtures


class PhoneImportTests(unittest.TestCase):
    def review_fixture(self, root):
        journal, model, runtime, video, manifest, rows = summary_test_fixtures.HypothesisSummaryTests().fixture(root,45)
        writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(96,64))
        self.assertTrue(writer.isOpened())
        for frame in range(45):
            image = np.full((64,96,3),frame*4,np.uint8)
            cv2.putText(image,str(frame+1),(5,40),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),1)
            writer.write(image)
        writer.release()
        manifest['source_sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
        manifest['source_fps'] = 30.
        for row in rows: row['source_time'] = (row['frame_id']-1)/30.
        helper = summary_test_fixtures.HypothesisSummaryTests()
        helper.write_json(journal/'manifest.json',manifest)
        helper.write_rows(journal,rows)
        _,selection = summarize(journal,model,runtime_path=runtime,max_bursts=3,session_id='session_1')
        selector = root/'selection.json'
        selector.write_text(json.dumps(selection,indent=2)+'\n')
        return video,root/'captures',selector,journal,selection

    def import_selected(self, video, output, selector, **kwargs):
        return import_recordings(video,output,'session_1',review_selection=selector,
                                 capture_sequence=2,capture_origin='unspecified',**kwargs)

    def test_sequential_frames_provenance_unknown_labels_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'originals'; source.mkdir(); output = root/'captures'
            video = source/'clip.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (96, 64))
            self.assertTrue(writer.isOpened())
            for i in range(45):
                image = np.full((64, 96, 3), i * 4, np.uint8)
                cv2.putText(image, str(i + 1), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 1)
                writer.write(image)
            writer.release()
            before = hashlib.sha256(video.read_bytes()).hexdigest()
            report = import_recordings(source, output, 'session_1', bursts=3)
            entry = report['recordings'][0]; clip = output/entry['clip']
            rows = [json.loads(line) for line in (clip/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(entry['source_frames'], 45); self.assertEqual(len(rows), 15)
            reference = cv2.VideoCapture(str(video)); decoded = []
            while True:
                ok, image = reference.read()
                if not ok: break
                decoded.append(image)
            reference.release()
            for row in rows:
                saved = cv2.imread(str(clip/row['file']))
                self.assertLess(np.abs(saved.astype(float) - decoded[row['source_frame'] - 1]).mean(), 3)
            annotations = json.loads((clip/'annotations.json').read_text())
            self.assertTrue(all(not label['reviewed'] for label in annotations.values()))
            annotations['frame_1.jpg']['reviewed'] = True
            (clip/'annotations.json').write_text(json.dumps(annotations))
            with self.assertRaises(FileExistsError): import_recordings(source, output, 'session_1')
            import_recordings(source, output, 'session_1', resume=True)
            self.assertTrue(json.loads((clip/'annotations.json').read_text())['frame_1.jpg']['reviewed'])
            with self.assertRaises(ValueError): import_recordings(source, output, 'another_session', resume=True)
            focused = import_recordings(video, output, 'session_1', bursts=3, start_frame=12,
                                        capture_sequence=2, note='Reported missed ball')
            focused_entry = focused['recordings'][0]; focused_dir = output/focused_entry['clip']
            focused_rows = [json.loads(line) for line in (focused_dir/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([row['source_frame'] for row in focused_rows], list(range(12, 27)))
            self.assertEqual(focused_entry['capture_note'], 'Reported missed ball')
            self.assertEqual(focused_entry['session_id'], entry['session_id'])
            self.assertIsNone(focused_entry['split'])
            self.assertTrue(all(not label['reviewed'] for label in json.loads((focused_dir/'annotations.json').read_text()).values()))
            for row in focused_rows:
                self.assertEqual(row['review_metadata']['reasons'], ['USER_REPORTED_DETECTION_FAILURE'])
                saved = cv2.imread(str(focused_dir/row['file']))
                self.assertLess(np.abs(saved.astype(float) - decoded[row['source_frame'] - 1]).mean(), 3)
            with self.assertRaises(ValueError):
                import_recordings(video, output, 'session_1', bursts=3, resume=True, start_frame=13, capture_sequence=2)
            with self.assertRaises(ValueError):
                import_recordings(video, output, 'session_1', resume=True, capture_sequence=2)
            with self.assertRaises(ValueError):
                import_recordings(video, output, 'session_1', bursts=3, start_frame=40, capture_sequence=3)
            self.assertFalse((output/('phone_'+before[:16]+'_3')).exists())
            self.assertTrue(json.loads((clip/'annotations.json').read_text())['frame_1.jpg']['reviewed'])
            self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(), before)

    def test_review_selected_frames_are_sequential_unreviewed_and_provenance_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            video,output,selector,journal,selection = self.review_fixture(Path(temporary))
            before = hashlib.sha256(video.read_bytes()).hexdigest()
            original = import_recordings(video,output,'session_1',bursts=2)
            original_dir = output/original['recordings'][0]['clip']
            original_labels = (original_dir/'annotations.json').read_bytes()
            real_capture = cv2.VideoCapture
            class SequentialOnly:
                def __init__(self,*args): self.capture = real_capture(*args)
                def set(self,*args): raise AssertionError('Random seeking must never occur')
                def __getattr__(self,name): return getattr(self.capture,name)
            with mock.patch.object(cv2,'VideoCapture',SequentialOnly):
                report = self.import_selected(video,output,selector,note='Suspected background failures')
            entry = report['recordings'][0]; clip = output/entry['clip']
            rows = [json.loads(line) for line in (clip/'frames.jsonl').read_text().splitlines()]
            expected = [frame for burst in selection['bursts'] for frame in range(burst['start_source_frame'],burst['end_source_frame']+1)]
            self.assertEqual([row['source_frame'] for row in rows],expected)
            self.assertEqual(entry['sampled_frames'],15)
            self.assertEqual(entry['capture_origin'],'unspecified')
            self.assertIsNone(entry['split']); self.assertFalse(entry['contains_ground_truth'])
            self.assertEqual(entry['selection']['selection_sha256'],hashlib.sha256(selector.read_bytes()).hexdigest())
            self.assertEqual(entry['selection']['identity'],selection['identity'])
            self.assertEqual(entry['selection']['bursts'],selection['bursts'])
            self.assertIn('cut/edit/duplicate history remains unverified',entry['capture_note'])
            self.assertFalse(entry['selection']['temporal_continuity_reviewed'])
            for row in rows:
                burst=selection['bursts'][row['burst_id']-1]
                self.assertEqual(row['review_metadata']['selection_burst'],burst)
                self.assertEqual(row['review_metadata']['reasons'],[burst['reason']])
                self.assertEqual(row['review_metadata']['selection_sha256'],entry['selection']['selection_sha256'])
            labels=json.loads((clip/'annotations.json').read_text())
            self.assertTrue(all(label=={'reviewed':False,'ball':[],'person':[],'table':[]} for label in labels.values()))
            reference=cv2.VideoCapture(str(video)); decoded=[]
            while True:
                ok,image=reference.read()
                if not ok: break
                decoded.append(image)
            reference.release()
            for row in rows:
                saved=cv2.imread(str(clip/row['file']))
                self.assertLess(np.abs(saved.astype(float)-decoded[row['source_frame']-1]).mean(),3)
            labels['frame_1.jpg']['reviewed']=True
            (clip/'annotations.json').write_text(json.dumps(labels))
            saved_labels=(clip/'annotations.json').read_bytes()
            self.import_selected(video,output,selector,resume=True)
            self.assertEqual((clip/'annotations.json').read_bytes(),saved_labels)
            self.assertEqual((original_dir/'annotations.json').read_bytes(),original_labels)
            self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(),before)
            with self.assertRaises(FileExistsError): self.import_selected(video,output,selector)
            with self.assertRaises(ValueError): import_recordings(video,output,'session_1',capture_sequence=2,capture_origin='unspecified',resume=True)
            selector.write_text(selector.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Existing review selection differs'):
                self.import_selected(video,output,selector,resume=True)
            self.assertEqual((clip/'annotations.json').read_bytes(),saved_labels)

    def test_review_selection_rejects_invalid_bounds_identity_and_conflicts(self):
        cases=('session','source_hash','source_path','count','overlap','duplicate','range','length','unsorted',
               'bad_schema','bad_method','labelled','unknown_truth','bad_anchor','bad_count','bad_reason',
               'missing_hash','missing_journal','stale_journal','nonfinite','empty')
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
                video,output,selector,journal,selection=self.review_fixture(Path(temporary))
                identity=selection['identity']; bursts=selection['bursts']
                if case=='session': selection['session_id']='other'
                elif case=='source_hash': identity['source_sha256']='0'*64
                elif case=='source_path': identity['source']='other.mp4'
                elif case=='count': identity['source_frame_count']=46
                elif case=='overlap': bursts[1]['start_source_frame']=bursts[0]['end_source_frame']; bursts[1]['end_source_frame']=bursts[1]['start_source_frame']+4
                elif case=='duplicate': bursts[1]=bursts[0]
                elif case=='range': bursts[-1]['end_source_frame']=46; bursts[-1]['start_source_frame']=42
                elif case=='length': bursts[0]['end_source_frame']+=1
                elif case=='unsorted': bursts.reverse()
                elif case=='bad_schema': selection['schema_version']=True
                elif case=='bad_method': selection['selection_method']='unknown'
                elif case=='labelled': selection['automatic_labels_created']=True
                elif case=='unknown_truth': selection['truth']['ground_truth_available']=True
                elif case=='bad_anchor': bursts[0]['anchor_center_normalized']=[.9,.9]
                elif case=='bad_count': selection['selected_source_frames']=14
                elif case=='bad_reason': bursts[0]['reason']='labelled'
                elif case=='missing_hash': identity.pop('journal_sha256')
                elif case=='missing_journal': (journal/'completion.json').unlink()
                elif case=='stale_journal': (journal/'completion.json').write_text('{}')
                elif case=='nonfinite': bursts[0]['anchor_timestamp_s']=float('nan')
                elif case=='empty': selection['bursts']=[]
                selector.write_text(json.dumps(selection))
                with self.assertRaises((ValueError,OSError)):
                    self.import_selected(video,output,selector)
                self.assertFalse(any(output.glob('phone_*')))
        with tempfile.TemporaryDirectory() as temporary:
            video,output,selector,_,_=self.review_fixture(Path(temporary))
            for kwargs in ({'start_frame':1},{'bursts':3}):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                    self.import_selected(video,output,selector,**kwargs)
            with self.assertRaises(ValueError):
                import_recordings(video.parent,output,'session_1',review_selection=selector,capture_sequence=2,capture_origin='unspecified')
            with self.assertRaises(ValueError):
                import_recordings(video,output,'session_1',review_selection=selector,capture_origin='unspecified')
            with self.assertRaises(ValueError):
                import_recordings(video,output,'session_1',review_selection=selector,capture_sequence=2)

    def test_review_selection_rechecks_journal_before_publish_and_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            video,output,selector,journal,_=self.review_fixture(Path(temporary))
            real_write=cv2.imwrite
            changed=False
            def mutate_evidence(*args):
                nonlocal changed
                result=real_write(*args)
                if not changed:
                    changed=True
                    with (journal/'frames.jsonl').open('ab') as stream: stream.write(b'\n')
                return result
            with mock.patch.object(cv2,'imwrite',mutate_evidence),self.assertRaisesRegex(ValueError,'stale'):
                self.import_selected(video,output,selector)
            self.assertFalse(any(output.glob('phone_*')))
        with tempfile.TemporaryDirectory() as temporary:
            video,output,selector,journal,_=self.review_fixture(Path(temporary))
            report=self.import_selected(video,output,selector)
            clip=output/report['recordings'][0]['clip']
            labels=(clip/'annotations.json').read_bytes()
            (journal/'frames.idx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'stale'):
                self.import_selected(video,output,selector,resume=True)
            self.assertEqual((clip/'annotations.json').read_bytes(),labels)

    def test_review_selection_must_match_actual_sequential_decode_count_and_fps(self):
        for case in ('count','fps'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                video,output,selector,journal,_=self.review_fixture(root)
                manifest=json.loads((journal/'manifest.json').read_text())
                rows=[json.loads(line) for line in (journal/'frames.jsonl').read_text().splitlines()]
                if case=='count':
                    rows=rows[:-1]
                    manifest['source_declared_frames']=manifest['total_frames']=44
                else:
                    manifest['source_fps']=25.
                    for row in rows: row['source_time']=(row['frame_id']-1)/25.
                helper=summary_test_fixtures.HypothesisSummaryTests()
                helper.write_json(journal/'manifest.json',manifest)
                helper.write_json(journal/'completion.json',{'status':'COMPLETE','analyzed_frames':len(rows),'total_frames':len(rows)})
                helper.write_rows(journal,rows)
                _,selection=summarize(journal,root/'model.engine',runtime_path=root/'runtime.exe',max_bursts=3,session_id='session_1')
                selector.write_text(json.dumps(selection))
                with self.assertRaisesRegex(ValueError,'decoded source'):
                    self.import_selected(video,output,selector)
                self.assertFalse(any(output.glob('phone_*')))


if __name__ == '__main__': unittest.main()
