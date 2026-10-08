import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from import_phone_recordings import import_recordings


class PhoneImportTests(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()
