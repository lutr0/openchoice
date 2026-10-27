"""User-visible supervision, leakage and probability-metric regressions."""
import json
from pathlib import Path
import tempfile
import unittest

from openchoice.finetune.data import convert, load_data, check_replay_leakage
from openchoice.finetune.training import summarize


class FineTuneDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = dict(id="one", state="Please refund a duplicate charge.",
            question={"type":"choice", "criteria":{"billing":"Billing", "access":"Login"}}, label="billing")

    def dataset(self, rows):
        path = self.root / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_label_types_are_not_coerced(self):
        for question, label in (({"type":"noul"}, "false"), ({"type":"score", "criteria":["low","high"]}, True), (self.base["question"], "missing")):
            with self.subTest(question=question), self.assertRaises(ValueError):
                convert(self.base | dict(question=question, label=label), self.root, 1)

    def test_soft_target_mass_and_binary_order(self):
        for probabilities in ([0.2, 0.2], [float('nan'), 0], [-0.1, 1.1], [1]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                convert(self.base | dict(probabilities=probabilities), self.root, 1)
        row = convert(self.base | dict(question={"type":"noul"}, label=False, probabilities=[0.2,0.8]), self.root, 1)
        self.assertEqual(row.row["target"], {"kind":"binary", "value":0.2})
        self.assertEqual(row.gold, 1)

    def test_cross_split_group_and_content_leakage_rejected(self):
        train = self.base | dict(split="train", group="conversation")
        dev = self.base | dict(id="two", split="dev", state="Cannot log in.", label="access", group="conversation")
        with self.assertRaisesRegex(ValueError, "group .* leaks"):
            load_data(self.dataset([train, dev]))
        dev = self.base | dict(id="two", split="dev")
        with self.assertRaisesRegex(ValueError, "duplicate decision .* leaks"):
            load_data(self.dataset([train, dev]))

    def test_replay_cannot_train_on_user_dev(self):
        user_dev = convert(self.base | dict(split="dev"), self.root, 1)
        replay_train = convert(self.base | dict(id="replay", split="train"), self.root, 2)
        with self.assertRaisesRegex(ValueError, "held-out decision"):
            check_replay_leakage([user_dev], [replay_train])

    def test_misspelled_noul_criterion_is_not_ignored(self):
        value = self.base | dict(question={"type":"noul", "criteria":{"tru":"verified"}}, label=True)
        with self.assertRaisesRegex(ValueError, "criteria keys"):
            convert(value, self.root, 1)

    def test_automatic_split_keeps_group_together_and_repeats(self):
        rows = [self.base | dict(id=f"row-{i}", group=f"thread-{i//2}", state=f"Invoice number {i}") for i in range(20)]
        path = self.dataset(rows)
        first, second = load_data(path, seed=123), load_data(path, seed=123)
        def assignment(data):
            return {e.row["id"]: e.row["split"] for e in data}
        self.assertEqual(assignment(first), assignment(second))
        for group in {e.row['group_id'] for e in first}:
            self.assertEqual(len({e.row['split'] for e in first if e.row['group_id'] == group}), 1)
        self.assertEqual({e.row['split'] for e in first}, {'train','dev'})

    def test_duplicate_keys_and_missing_split_report_errors(self):
        path = self.root / "bad.jsonl"
        path.write_text('{"state":"a","state":"b"}\n')
        with self.assertRaisesRegex(ValueError, "bad.jsonl:1: Duplicate"):
            load_data(path)
        rows = [self.base | dict(split="train"), self.base | dict(id="two", state="Other issue")]
        with self.assertRaisesRegex(ValueError, "every row"):
            load_data(self.dataset(rows))

    def test_score_levels_preserve_order(self):
        example = convert(self.base | dict(question={"type":"score", "criteria":["critical", "ordinary", "minor"]}, label=0), self.root, 1)
        self.assertEqual(example.row['target'], {'kind':'class','id':'0'})
        self.assertEqual([o['text'] for o in example.row['options']], ['critical','ordinary','minor'])

    def test_image_path_cannot_escape_dataset(self):
        with self.assertRaisesRegex(ValueError, "relative path"):
            convert(self.base | dict(image="../private.png"), self.root, 1)

    def test_metric_aggregation_uses_decision_count(self):
        records = [dict(accuracy=1, brier=0.02, log_loss=0.1), dict(accuracy=0, brier=1.8, log_loss=2.4)]
        self.assertEqual(summarize(records), dict(rows=2, accuracy=0.5, brier=0.91, log_loss=1.25))
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize([])


if __name__ == '__main__':
    unittest.main()
