from nanotrack.models.head import ban_v2, ban_v3


def get_ban_head(name, version="v2", **kwargs):
    if name not in {"UPChannelBAN", "DepthwiseBAN"}:
        raise KeyError(name)
    module = ban_v3 if str(version).lower() == "v3" else ban_v2
    return getattr(module, name)(**kwargs)
