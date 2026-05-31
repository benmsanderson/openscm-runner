"""
Settings handling

Rather than hard-coding constants, configuration can be source from 2 different
sources:

* Environment variables
* dotenv files

Environment variables are always used in preference to the values in dotenv
files.
"""


import logging
import multiprocessing
import os

from dotenv import dotenv_values, find_dotenv

LOGGER = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration container

    Loads a local dotenv file containing configuration. An example of a
    configuration file is provided in root of the project.

    A combination of environment variables and dotenv files can be used as
    configuration with an existing environment variables overriding a value
    specified in a dotenv file.

    All configuration is case-insensitive.

    .. code:: python

        >>> config = ConfigLoader()
        >>> config["VALUE"]
        Traceback (most recent call last):
         ...
        KeyError: 'VALUE'
        >>> config.get("VALUE", "some_default")
        'some_default'
    """

    def __init__(self):
        self._config = {}
        self.is_loaded = False

    def load_config(self):
        """
        Load configuration from disk

        A dotenv file is attempted to be loaded. The first file named .env
        in the current directory or any of its parents will read in.

        If no dotenv files are found, then
        """
        dotenv_cfg = dotenv_values(find_dotenv(usecwd=True), verbose=True)
        self.update(dotenv_cfg)

        # Add any extra files here

        self.is_loaded = True

    def get(self, key, default=None):
        """
        Get value for a given key, falling back to a default value if not present.

        Parameters
        ----------
        key : str
            Key

        default: Any
            Default value returned if no configuration for ``key`` is present.

        Returns
        -------
        Any
            Value with key ``item``. If not value is present ``default``
            is returned.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        """
        Get a config value for a given key

        Parameters
        ----------
        key: str
            Key

        Returns
        -------
        Any
            Value with key ``item``

        Raises
        ------
        KeyError
            No configuration values for the key were found
        """
        if not self.is_loaded:
            # Lazy loaded
            self.load_config()
        key = key.upper()

        # Always preference environment variable
        if key in os.environ:
            return os.environ[key]

        # Fall back to loaded config
        # Preference determined by load order
        if key in self._config:
            return self._config[key]

        # Raise KeyError if not found
        raise KeyError(key)

    def update(self, conf):
        """
        Update the configuration

        If configuration with duplicate keys already exists, then these values
        will overwrite the existing values.

        Parameters
        ----------
        conf: dict
            Configuration to use to update the config
        """
        conf = {k.upper(): v for k, v in conf.items()}
        self._config.update(conf)


config = ConfigLoader()


def get_worker_count(override_env_var):
    """
    Resolve the number of worker processes to use, accounting for cluster
    CPU allocations.

    Lookup order:

    1. The adapter-specific override (``override_env_var``), e.g.
       ``"FAIR_WORKER_NUMBER"``. Read via the shared :data:`config`
       loader so it can come from either the environment or a dotenv
       file. If set, returned as-is; users keep full control.
    2. ``SLURM_CPUS_PER_TASK`` if set in the environment (SLURM exports
       this for the current job allocation, so it reflects what the
       scheduler actually gave us rather than the whole physical node).
    3. ``OMP_NUM_THREADS`` if set (general-purpose thread-count hint).
    4. :func:`multiprocessing.cpu_count` as a final fallback (entire
       host, the pre-existing default).

    Note that when top-level ``parallel_models`` dispatch is active in
    :func:`openscm_runner.run.run`, multiple adapters may each spawn a
    pool of this size concurrently. To avoid oversubscription on a
    shared node, set the adapter-specific override env vars (e.g.
    ``FAIR_WORKER_NUMBER``, ``MAGICC_WORKER_NUMBER``,
    ``CICEROSCM_WORKER_NUMBER``) so the aggregate fits the allocation.

    Parameters
    ----------
    override_env_var : str
        Name of the adapter-specific override env var, e.g.
        ``"FAIR_WORKER_NUMBER"``.

    Returns
    -------
    int
        Worker count to use, always at least 1.
    """
    for env_var in (override_env_var, "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = config.get(env_var)
        if value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Ignoring non-integer worker count from %s: %r", env_var, value
            )
            continue
        return max(count, 1)
    return multiprocessing.cpu_count()
