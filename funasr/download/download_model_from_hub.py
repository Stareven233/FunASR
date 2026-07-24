import logging
import os
import json
from omegaconf import OmegaConf, DictConfig

from funasr.download.name_maps_from_hub import name_maps_ms, name_maps_hf, name_maps_openai


def _cache_miss_error(hub: str, model_id: str, *, cache_dir=None, local_files_only=False, cause=None):
    """Build an actionable error for a missing / unreadable hub cache entry."""
    cache_msg = f"cache_dir={cache_dir!r}" if cache_dir else "cache_dir=<default>"
    hint = (
        "Re-run with --allow-download (or set local_files_only=False) to fetch missing artifacts."
        if local_files_only
        else "Check network access and model id, or pass a local model directory."
    )
    msg = (
        f"Failed to resolve model from {hub}: {model_id!r} "
        f"({cache_msg}, local_files_only={local_files_only}). {hint}"
    )
    if cause is not None:
        msg = f"{msg} Original error: {cause}"
    return RuntimeError(msg)


def download_model(**kwargs):
    """Download model from hub and parse its configuration.

    Resolves model name aliases, downloads from ModelScope or HuggingFace,
    reads config.yaml and configuration.json, and returns complete kwargs
    for model instantiation.

    Args:
        **kwargs: Must include 'model' (str). Optional: 'hub', 'model_revision',
            'cache_dir', 'local_files_only', 'check_latest', 'is_training', etc.

    Returns:
        dict: Complete kwargs with resolved paths, model class name, and config.
    """
    hub = kwargs.get("hub", "ms")
    if hub == "ms" or hub == "modelscope":
        kwargs = download_from_ms(**kwargs)
    elif hub == "hf" or hub == "huggingface":
        kwargs = download_from_hf(**kwargs)
    elif hub == "openai":
        model_or_path = kwargs.get("model")
        if os.path.exists(model_or_path):
            # local path
            kwargs["model_path"] = model_or_path
            kwargs["model"] = "WhisperWarp"
        else:
            # model name
            if model_or_path in name_maps_openai:
                model_or_path = name_maps_openai[model_or_path]
            kwargs["model_path"] = model_or_path

    return kwargs


def download_from_ms(**kwargs):
    """Download from ms.

        Args:
            **kwargs: Additional keyword arguments.
        """
    model_or_path = kwargs.get("model")
    if model_or_path in name_maps_ms:
        model_or_path = name_maps_ms[model_or_path]
    model_revision = kwargs.get("model_revision", "master")
    cache_dir = kwargs.get("cache_dir", None)
    local_files_only = bool(kwargs.get("local_files_only", False))
    check_latest = kwargs.get("check_latest", True)
    if not os.path.exists(model_or_path) and "model_path" not in kwargs:
        try:
            model_or_path = get_or_download_model_dir(
                model_or_path,
                model_revision,
                is_training=kwargs.get("is_training"),
                check_latest=check_latest,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as e:
            raise _cache_miss_error(
                "ModelScope",
                model_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                cause=e,
            ) from e
        if not os.path.exists(model_or_path):
            raise _cache_miss_error(
                "ModelScope",
                kwargs.get("model"),
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

    kwargs["model_path"] = model_or_path if "model_path" not in kwargs else kwargs["model_path"]
    model_or_path = kwargs["model_path"]
    if os.path.exists(os.path.join(model_or_path, "configuration.json")):
        with open(os.path.join(model_or_path, "configuration.json"), "r", encoding="utf-8") as f:
            conf_json = json.load(f)

            cfg = {}
            if "file_path_metas" in conf_json:
                add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
            # cfg.update(kwargs)
            cfg = OmegaConf.merge(cfg, kwargs)
            if "config" in cfg:
                config = OmegaConf.load(cfg["config"])
                kwargs = OmegaConf.merge(config, cfg)
                kwargs["model"] = config["model"]
    elif os.path.exists(os.path.join(model_or_path, "config.yaml")):
        config = OmegaConf.load(os.path.join(model_or_path, "config.yaml"))
        kwargs = OmegaConf.merge(config, kwargs)
        init_param = os.path.join(model_or_path, "model.pt")
        if "init_param" not in kwargs or not os.path.exists(kwargs["init_param"]):
            kwargs["init_param"] = init_param
            assert os.path.exists(kwargs["init_param"]), "init_param does not exist"
        if os.path.exists(os.path.join(model_or_path, "tokens.txt")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.txt")
        if os.path.exists(os.path.join(model_or_path, "tokens.json")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.json")
        if os.path.exists(os.path.join(model_or_path, "seg_dict")):
            kwargs["tokenizer_conf"]["seg_dict"] = os.path.join(model_or_path, "seg_dict")
        if os.path.exists(os.path.join(model_or_path, "bpe.model")):
            kwargs["tokenizer_conf"]["bpemodel"] = os.path.join(model_or_path, "bpe.model")
        kwargs["model"] = config["model"]
        if os.path.exists(os.path.join(model_or_path, "am.mvn")):
            kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
        if os.path.exists(os.path.join(model_or_path, "jieba_usr_dict")):
            kwargs["jieba_usr_dict"] = os.path.join(model_or_path, "jieba_usr_dict")
    if isinstance(kwargs, DictConfig):
        kwargs = OmegaConf.to_container(kwargs, resolve=True)
    logging.warning(f'trust_remote_code: {kwargs.get("trust_remote_code", False)}')
    if os.path.exists(os.path.join(model_or_path, "requirements.txt")) and kwargs.get(
        "trust_remote_code", False
    ):
        requirements = os.path.join(model_or_path, "requirements.txt")
        print(f"Detect model requirements, begin to install it: {requirements}")
        from funasr.utils.install_model_requirements import install_requirements

        install_requirements(requirements)
    if kwargs.get("trust_remote_code", False):
        from funasr.utils.dynamic_import import import_module_from_path

        model_code = kwargs.get("remote_code", "model")
        import_module_from_path(model_code)

        # from funasr.register import tables
        # tables.print("model")
    return kwargs


def download_from_hf(**kwargs):
    """Download from hf.

        Args:
            **kwargs: Additional keyword arguments.
        """
    model_or_path = kwargs.get("model")
    if model_or_path in name_maps_hf:
        model_or_path = name_maps_hf[model_or_path]
    model_revision = kwargs.get("model_revision", "master")
    cache_dir = kwargs.get("cache_dir", None)
    local_files_only = bool(kwargs.get("local_files_only", False))
    check_latest = kwargs.get("check_latest", True)
    if not os.path.exists(model_or_path) and "model_path" not in kwargs:
        try:
            model_or_path = get_or_download_model_dir_hf(
                model_or_path,
                model_revision,
                is_training=kwargs.get("is_training"),
                check_latest=check_latest,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as e:
            raise _cache_miss_error(
                "HuggingFace",
                model_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                cause=e,
            ) from e
        if not os.path.exists(model_or_path):
            raise _cache_miss_error(
                "HuggingFace",
                kwargs.get("model"),
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

    kwargs["model_path"] = model_or_path if "model_path" not in kwargs else kwargs["model_path"]

    if os.path.exists(os.path.join(model_or_path, "configuration.json")):
        with open(os.path.join(model_or_path, "configuration.json"), "r", encoding="utf-8") as f:
            conf_json = json.load(f)

            cfg = {}
            if "file_path_metas" in conf_json:
                add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
            cfg.update(kwargs)
            if "config" in cfg:
                config = OmegaConf.load(cfg["config"])
                kwargs = OmegaConf.merge(config, cfg)
                kwargs["model"] = config["model"]
    elif os.path.exists(os.path.join(model_or_path, "config.yaml")) and os.path.exists(
        os.path.join(model_or_path, "model.pt")
    ):
        config = OmegaConf.load(os.path.join(model_or_path, "config.yaml"))
        kwargs = OmegaConf.merge(config, kwargs)
        init_param = os.path.join(model_or_path, "model.pt")
        kwargs["init_param"] = init_param
        if os.path.exists(os.path.join(model_or_path, "tokens.txt")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.txt")
        if os.path.exists(os.path.join(model_or_path, "tokens.json")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.json")
        if os.path.exists(os.path.join(model_or_path, "seg_dict")):
            kwargs["tokenizer_conf"]["seg_dict"] = os.path.join(model_or_path, "seg_dict")
        if os.path.exists(os.path.join(model_or_path, "bpe.model")):
            kwargs["tokenizer_conf"]["bpemodel"] = os.path.join(model_or_path, "bpe.model")
        kwargs["model"] = config["model"]
        if os.path.exists(os.path.join(model_or_path, "am.mvn")):
            kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
        if os.path.exists(os.path.join(model_or_path, "jieba_usr_dict")):
            kwargs["jieba_usr_dict"] = os.path.join(model_or_path, "jieba_usr_dict")
    if isinstance(kwargs, DictConfig):
        kwargs = OmegaConf.to_container(kwargs, resolve=True)
    logging.warning(f'trust_remote_code: {kwargs.get("trust_remote_code", False)}')
    if os.path.exists(os.path.join(model_or_path, "requirements.txt")) and kwargs.get(
        "trust_remote_code", False
    ):
        requirements = os.path.join(model_or_path, "requirements.txt")
        print(f"Detect model requirements, begin to install it: {requirements}")
        from funasr.utils.install_model_requirements import install_requirements

        install_requirements(requirements)
    return kwargs


def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg={}):
    """Add file root path.

        Args:
            model_or_path: TODO.
            file_path_metas: TODO.
            cfg: Configuration overrides.
        """
    if isinstance(file_path_metas, dict):
        if isinstance(cfg, list):
            cfg.append({})

        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    if isinstance(cfg, dict):
                        cfg[k] = p
                    elif isinstance(cfg, list):
                        # if len(cfg) == 0:
                        # cfg.append({})
                        cfg[-1][k] = p

            elif isinstance(v, dict):
                if isinstance(cfg, dict):
                    if k not in cfg:
                        cfg[k] = {}
                    add_file_root_path(model_or_path, v, cfg[k])
                # elif isinstance(cfg, list):
                #     cfg.append({})
                #     add_file_root_path(model_or_path, v, cfg)
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if k not in cfg:
                        cfg[k] = []
                    if isinstance(vv, str):
                        p = os.path.join(model_or_path, vv)
                        # file_path_metas[i] = p
                        if os.path.exists(p):
                            if isinstance(cfg[k], dict):
                                cfg[k] = p
                            elif isinstance(cfg[k], list):
                                cfg[k].append(p)
                    elif isinstance(vv, dict):
                        add_file_root_path(model_or_path, vv, cfg[k])

    return cfg


def get_or_download_model_dir(
    model,
    model_revision=None,
    is_training=False,
    check_latest=True,
    cache_dir=None,
    local_files_only=False,
):
    """Get local model directory or download model if necessary.

    Args:
        model (str): model id or path to local model directory.
        model_revision  (str, optional): model version number.
        is_training (bool): whether invoked from trainer.
        check_latest (bool): when True and model is a local path, optionally
            contact ModelScope to verify the cache is up to date. Default True
            preserves library online behavior; CLI scripts should pass False.
        cache_dir (str, optional): ModelScope cache root override.
        local_files_only (bool): if True, never contact the hub; raise on miss.
    """
    from modelscope.hub.check_model import check_local_model_is_latest
    from modelscope.hub.snapshot_download import snapshot_download

    from modelscope.utils.constant import Invoke, ThirdParty

    key = Invoke.LOCAL_TRAINER if is_training else Invoke.PIPELINE

    if os.path.exists(model):
        # Local filesystem path: return it directly. Only contact the hub for a
        # freshness check when explicitly requested and not in local-only mode.
        model_cache_dir = model if os.path.isdir(model) else os.path.dirname(model)
        if check_latest and not local_files_only:
            try:
                check_local_model_is_latest(
                    model_cache_dir, user_agent={Invoke.KEY: key, ThirdParty.KEY: "funasr"}
                )
            except Exception:
                print("could not check the latest version")
        return model_cache_dir

    snap_kwargs = {
        "revision": model_revision,
        "user_agent": {Invoke.KEY: key, ThirdParty.KEY: "funasr"},
        "local_files_only": bool(local_files_only),
    }
    if cache_dir is not None:
        snap_kwargs["cache_dir"] = cache_dir
    model_cache_dir = snapshot_download(model, **snap_kwargs)
    return model_cache_dir


def get_or_download_model_dir_hf(
    model,
    model_revision=None,
    is_training=False,
    check_latest=True,
    cache_dir=None,
    local_files_only=False,
):
    """Get local model directory or download model if necessary.

    Args:
        model (str): model id or path to local model directory.
        model_revision  (str, optional): model version number. The historical
            default "master" is treated as None so HuggingFace uses the repo
            default branch (typically "main").
        is_training (bool): unused; kept for signature parity with MS helper.
        check_latest (bool): unused for HF (no equivalent remote freshness
            check); accepted for API symmetry.
        cache_dir (str, optional): HuggingFace hub cache root override.
        local_files_only (bool): if True, never contact the hub; raise on miss.
    """
    from huggingface_hub import snapshot_download

    if os.path.exists(model):
        return model if os.path.isdir(model) else os.path.dirname(model)

    # FunASR historically defaulted model_revision to "master" (ModelScope).
    # HuggingFace repos use "main"; pass None so the hub picks the default.
    revision = model_revision
    if revision in (None, "", "master"):
        revision = None

    snap_kwargs = {
        "local_files_only": bool(local_files_only),
    }
    if revision is not None:
        snap_kwargs["revision"] = revision
    if cache_dir is not None:
        snap_kwargs["cache_dir"] = cache_dir
    # check_latest has no HF equivalent; ignored intentionally.
    _ = (is_training, check_latest)
    model_cache_dir = snapshot_download(model, **snap_kwargs)
    return model_cache_dir
