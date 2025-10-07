import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from session import SessionManager
import pytest

def test_append_and_retrieve():
    sm = SessionManager()
    sm.start_session("s1")
    sm.append_user_turn("s1", "Hi")
    sm.append_assistant_turn("s1", "Hello")

    history = sm.get_history("s1")
    assert len(history) == 2
    assert history[0]["speaker"] == "user"
    assert history[0]["text"] == "Hi"
    assert history[1]["speaker"] == "assistant"
    assert history[1]["text"] == "Hello"

def test_append_without_start():
    sm = SessionManager()
    sm.append_user_turn("s2", "Question?")
    history = sm.get_history("s2")
    assert len(history) == 1
    assert history[0]["speaker"] == "user"

def test_clear_session():
    sm = SessionManager()
    sm.append_user_turn("s3", "Hi")
    sm.clear_session("s3")
    history = sm.get_history("s3")
    assert history == []
