# coding: utf-8
"""The decision-model layer's value objects and the invariants ``validate_answers`` enforces on them."""

from __future__ import annotations

import base64
from typing import Any
from unittest import TestCase

from openjiuwen.core.common.exception.errors import BaseError

from s1a.decision_models import (
    Choice,
    ChoiceQuestion,
    Decision,
    Image,
    Noul,
    NoulQuestion,
    Observation,
    Reply,
    Usage,
    choice_faults,
    validate_answers,
    validate_choice,
    validate_noul,
)

IDS = ["inc", "noop"]
GOOD = Choice(key="inc", probabilities={"inc": 0.7, "noop": 0.3}, confidence=0.7)
GOOD_FIELDS = GOOD.as_dict()


def _fields(key: Any, probabilities: Any, confidence: Any) -> dict[str, Any]:
    return {"choice": key, "probabilities": probabilities, "confidence": confidence}


class TestObservationAndQuestions(TestCase):
    def test_an_observation_is_a_dict_or_a_string_with_images_as_a_tuple(self) -> None:
        self.assertEqual(Observation({"n": 1}).images, ())
        self.assertEqual(Observation("plain text").state, "plain text")
        image = Image.from_base64(base64.b64encode(b"png").decode())
        self.assertEqual((image.data, image.media_type), (b"png", "image/png"))
        self.assertEqual(Observation({}, images=[image]).images, (image,))
        with self.assertRaises(TypeError):
            Observation([1, 2])  # type: ignore[arg-type]

    def test_a_choice_question_needs_options_with_string_keys_and_takes_one_rule_as_a_string(self) -> None:
        question = ChoiceQuestion({"a": "one", "b": {"element": "[2] Where to?"}}, rules="stand on 17")
        self.assertEqual((question.type, question.ids, question.rules), ("choice", ["a", "b"], ("stand on 17",)))
        self.assertEqual(ChoiceQuestion({"a": ""}, rules=["r1", "r2"]).rules, ("r1", "r2"))
        self.assertEqual(ChoiceQuestion({"a": ""}).rules, ())
        with self.assertRaises(ValueError):
            ChoiceQuestion({})
        with self.assertRaises(ValueError):
            ChoiceQuestion({1: "one"})  # type: ignore[dict-item]

    def test_a_noul_question_needs_text_and_true_false_criteria(self) -> None:
        question = NoulQuestion("Does it hold?", {"true": "yes", "false": "no"})
        self.assertEqual((question.type, question.criteria), ("noul", {"true": "yes", "false": "no"}))
        self.assertIsNone(NoulQuestion("Does it hold?").criteria)
        with self.assertRaises(ValueError):
            NoulQuestion(" ")
        with self.assertRaises(ValueError):
            NoulQuestion("q", {"yes": "y", "no": "n"})


class TestAnswersAndUsage(TestCase):
    def test_as_dict_is_the_cli_shape(self) -> None:
        self.assertEqual(
            GOOD.as_dict(), {"choice": "inc", "probabilities": {"inc": 0.7, "noop": 0.3}, "confidence": 0.7}
        )
        self.assertEqual(Noul(0.9, 0.9).as_dict(), {"noul": 0.9, "confidence": 0.9})

    def test_usage_tolerates_a_missing_or_malformed_payload(self) -> None:
        self.assertEqual(Usage.from_payload({"input_tokens": 315, "output_tokens": 31}), Usage(315, 31))
        self.assertEqual(Usage.from_payload({"input_tokens": 0}), Usage(0, 0, known=True), "an explicit zero is known")
        self.assertEqual(Usage.from_payload({"input_tokens": 4, "output_tokens": 0}), Usage(4, 0, known=True))
        # Missing or malformed usage is unknown, not a confirmed zero; the tolerant counts are still kept.
        self.assertEqual(Usage.from_payload(None), Usage(0, 0, known=False))
        self.assertEqual(Usage.from_payload("junk"), Usage(0, 0, known=False))
        self.assertEqual(Usage.from_payload({}), Usage(0, 0, known=False), "input must be present")
        self.assertEqual(Usage.from_payload({"output_tokens": 7}), Usage(0, 7, known=False))
        self.assertEqual(Usage.from_payload({"input_tokens": "x", "output_tokens": None}), Usage(0, 0, known=False))
        for bad in (True, -1, 1.5, "3", float("inf")):
            with self.subTest(bad=bad):
                self.assertFalse(Usage.from_payload({"input_tokens": bad}).known)
        self.assertFalse(Usage.from_payload({"input_tokens": 5, "output_tokens": -1}).known)

    def test_a_decision_hands_out_answers_by_type(self) -> None:
        decision = Decision(answers={"pick": GOOD, "check": Noul(0.2, 0.8)}, latency_ms=9)
        self.assertIs(decision.choice("pick"), GOOD)
        self.assertEqual(decision.noul("check").p, 0.2)
        with self.assertRaises(TypeError):
            decision.choice("check")
        with self.assertRaises(TypeError):
            decision.noul("pick")
        with self.assertRaises(KeyError):
            decision.choice("missing")


class TestValidation(TestCase):
    def test_a_good_choice_passes_with_its_numbers_normalised(self) -> None:
        choice = validate_choice(_fields("inc", {"inc": 1, "noop": 0}, 1), IDS)
        self.assertEqual(choice, Choice("inc", {"inc": 1.0, "noop": 0.0}, 1.0))
        self.assertEqual(choice_faults(GOOD_FIELDS, IDS), [])

    def test_every_fault_is_named(self) -> None:
        cases = {
            "not one of": _fields("dec", {"inc": 1.0, "noop": 0.0}, 1.0),
            "cover": _fields("inc", {"inc": 1.0}, 1.0),
            "sum to": _fields("inc", {"inc": 0.3, "noop": 0.3}, 0.9),
            "not the peak": _fields("inc", {"inc": 0.3, "noop": 0.7}, 0.7),
            "confidence": _fields("inc", {"inc": 1.0, "noop": 0.0}, 1.5),
            "not an object": _fields("inc", None, 1.0),
            "not in [0, 1]": _fields("inc", {"inc": 1.2, "noop": -0.2}, 1.0),
            "choice None": {},
            "choice ['inc']": _fields(["inc"], {"inc": 1.0, "noop": 0.0}, 1.0),
            "choice {'key': 'inc'}": _fields({"key": "inc"}, {"inc": 1.0, "noop": 0.0}, 1.0),
        }
        for fragment, fields in cases.items():
            with self.subTest(fragment=fragment):
                faults = choice_faults(fields, IDS)
                self.assertTrue(any(fragment in fault for fault in faults), faults)
                with self.assertRaises(BaseError) as caught:
                    validate_choice(fields, IDS)
                self.assertIn("not an offered choice", str(caught.exception))
                self.assertIn(fragment, str(caught.exception))

    def test_ties_at_the_peak_and_a_sum_within_two_percent_are_accepted(self) -> None:
        self.assertEqual(choice_faults(_fields("inc", {"inc": 0.5, "noop": 0.5}, 0.5), IDS), [])
        self.assertEqual(choice_faults(_fields("inc", {"inc": 0.6, "noop": 0.39}, 0.5), IDS), [])

    def test_a_noul_needs_a_probability_and_defaults_its_confidence(self) -> None:
        self.assertEqual(validate_noul({"noul": 0.2}), Noul(0.2, 0.8))
        self.assertEqual(validate_noul({"noul": 0.9, "confidence": 0.55}), Noul(0.9, 0.55))
        self.assertEqual(validate_noul({"noul": 0.9, "confidence": 7}).confidence, 0.9)
        for bad in (1.7, -0.1, None, "0.5", True):
            with self.assertRaises(BaseError):
                validate_noul({"noul": bad})
        with self.assertRaises(BaseError):
            validate_noul({})

    def test_validate_answers_builds_the_decision_from_every_asked_question(self) -> None:
        questions = {"pick": ChoiceQuestion({"inc": "", "noop": ""}), "check": NoulQuestion("q")}
        reply = Reply(
            answers={"pick": GOOD_FIELDS, "check": {"noul": 0.2}, "extra": GOOD_FIELDS},
            latency_ms=3,
            usage=Usage(5, 1),
            model="m",
            raw={"answers": "whatever came"},
        )
        decision = validate_answers(reply, questions)
        self.assertEqual(sorted(decision.answers), ["check", "pick"])
        self.assertEqual((decision.choice("pick"), decision.noul("check")), (GOOD, Noul(0.2, 0.8)))
        self.assertEqual(
            (decision.latency_ms, decision.usage, decision.model, decision.raw), (3, Usage(5, 1), "m", reply.raw)
        )
        with self.assertRaises(BaseError) as caught:
            validate_answers(Reply(answers={"pick": GOOD_FIELDS}, latency_ms=0), questions)
        self.assertIn("no answer for question 'check'", str(caught.exception))

    def test_an_answer_of_the_wrong_shape_or_no_object_fails_on_its_fields(self) -> None:
        questions = {"pick": ChoiceQuestion({"inc": "", "noop": ""}), "check": NoulQuestion("q")}
        swapped = Reply(answers={"pick": {"noul": 0.5}, "check": GOOD_FIELDS}, latency_ms=0)
        with self.assertRaises(BaseError) as caught:
            validate_answers(swapped, questions)
        self.assertIn("not an offered choice", str(caught.exception))
        for junk in ("nonsense", None, [1]):
            with self.assertRaises(BaseError):
                validate_answers(Reply(answers={"pick": junk, "check": {"noul": 0.5}}, latency_ms=0), questions)
