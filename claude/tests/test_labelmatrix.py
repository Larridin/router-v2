import json

from routerlab.labelmatrix import extract_answer, extract_code, grade_mbpp, grade_mcq, grade_number


def test_extract_answer_takes_last_occurrence():
    assert extract_answer("The answer: B is tempting but...\nAnswer: C") == "C"
    assert extract_answer("blah **Answer: 42**") == "42"
    assert extract_answer("Answer: 1,234") == "1234"
    assert extract_answer("no verdict here") == ""


def test_grade_mcq():
    assert grade_mcq("reasoning... Answer: D", "D") == 1.0
    assert grade_mcq("Answer: A", "d") == 0.0


def test_grade_number():
    assert grade_number("steps... Answer: 42", "42") == 1.0
    assert grade_number("Answer: 42.0", "42") == 1.0
    assert grade_number("Answer: 41", "42") == 0.0
    assert grade_number("gibberish", "42") == 0.0


def test_extract_code_prefers_last_block():
    text = "```python\nx=1\n```\ntext\n```python\ndef f():\n    return 2\n```"
    assert "return 2" in extract_code(text)


def test_grade_mbpp_executes_tests():
    gold = json.dumps({"tests": ["assert add(2, 3) == 5"], "setup": ""})
    assert grade_mbpp("```python\ndef add(a, b):\n    return a + b\n```", gold) == 1.0
    assert grade_mbpp("```python\ndef add(a, b):\n    return a - b\n```", gold) == 0.0
    assert grade_mbpp("no code at all", gold) == 0.0
