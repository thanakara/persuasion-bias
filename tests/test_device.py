from pytest_mock import MockerFixture

from persuasion_bias.utils.device import decide_device


def test_decide_device_cuda(mocker: MockerFixture):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=False)

    assert decide_device() == "cuda"


def test_decide_device_mps(mocker: MockerFixture):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=True)

    assert decide_device() == "mps"


def test_decide_device_cpu(mocker: MockerFixture):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)

    assert decide_device() == "cpu"
