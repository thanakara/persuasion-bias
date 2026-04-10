from omegaconf import OmegaConf
from pytest_mock import MockerFixture

from persuasion_bias.utils.hydra import load_config, setup_logging, register_resolvers


def test_register_resolvers(mocker: MockerFixture):
    """Check resolvers are registered without errors."""
    mocker.patch.object(OmegaConf, "register_new_resolver")
    register_resolvers()

    assert OmegaConf.register_new_resolver.call_count == 2  # noqa: PLR2004


def test_setup_logging(mocker: MockerFixture):
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data="version: 1"))
    mock_dict_config = mocker.patch("logging.config.dictConfig")
    mocker.patch("yaml.safe_load", return_value={"version": 1})

    setup_logging()

    mock_open.assert_called_once()
    mock_dict_config.assert_called_once_with({"version": 1})


def test_load_config(mocker: MockerFixture):
    mock_cfg = OmegaConf.create({"hydra": {}, "model": ""})

    mocker.patch("persuasion_bias.utils.hydra.setup_logging")
    mocker.patch("persuasion_bias.utils.hydra.initialize_config_dir")
    mocker.patch("persuasion_bias.utils.hydra.compose", return_value=mock_cfg)
    mocker.patch("persuasion_bias.utils.hydra.HydraConfig.instance")

    cfg = load_config()

    assert cfg is not None


def test_load_config_skips_logging(mocker: MockerFixture):
    mock_cfg = OmegaConf.create({"hydra": {}, "model": ""})
    mock_setup_log = mocker.patch("persuasion_bias.utils.hydra.setup_logging")

    mocker.patch("persuasion_bias.utils.hydra.initialize_config_dir")
    mocker.patch("persuasion_bias.utils.hydra.compose", return_value=mock_cfg)
    mocker.patch("persuasion_bias.utils.hydra.HydraConfig.instance")

    load_config(setup_log=False)

    mock_setup_log.assert_not_called()
