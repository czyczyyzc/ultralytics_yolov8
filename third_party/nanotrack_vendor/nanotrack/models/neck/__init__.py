from nanotrack.models.neck.neck import AdjustAllLayer, AdjustLayer

NECKS = {"AdjustLayer": AdjustLayer, "AdjustAllLayer": AdjustAllLayer}


def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
