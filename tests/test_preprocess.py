import pytest


def test_placeholder():
    pytest.skip("待实现")


@pytest.mark.skip(reason="video reserved")
def test_video_loader_reserved():
    pass
