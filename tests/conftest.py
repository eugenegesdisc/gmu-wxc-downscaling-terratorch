import pytest
from dataclasses import dataclass

@dataclass
class TestWithMarkSkipper:
    """Util to skip tests with mark, unless cli option provided."""

    test_mark: str
    cli_option_name: str
    cli_option_help: str

    def pytest_addoption_hook(self, parser):
        parser.addoption(
            self.cli_option_name,
            action="store_true",
            default=False,
            help=self.cli_option_help
        )
        # other argument options
        parser.addoption(
            "--earthdata-login-config",
            action="store",
            default="../cfg/.gmu_downscaling_earthdata_login.cfg",
            help="Location of the configuration file to retrieve username/password or token for earthdata-login"
        )
    
    def pytest_collection_modifyitems_hook(self, config, items):
        if not config.getoption(self.cli_option_name):
            self._skip_items_with_mark(items)

    def _skip_items_with_mark(self, items):
        reason = f"need {self.cli_option_name} option to run"
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if self.test_mark in item.keywords:
                item.add_marker(skip_marker)
    

optional_skipper = TestWithMarkSkipper(
    test_mark="option1",
    cli_option_name="--runoption1",
    cli_option_help="run option1 set of tests"
)

pytest_addoption = optional_skipper.pytest_addoption_hook
pytest_collection_modifyitems = optional_skipper.pytest_collection_modifyitems_hook
