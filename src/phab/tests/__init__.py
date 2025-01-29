import pytest


@pytest.fixture
def somethingThatDoesntExist() -> str:
    return "something-that-does-not-exist-ololo"
