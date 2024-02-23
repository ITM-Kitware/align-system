# Borrowed from: https://github.com/NextCenturyCorporation/itm-evaluation-server/blob/development/swagger_server/util.py
def get_swagger_class_enum_values(klass):
    return [getattr(klass, i) for i in dir(klass)
            if not i.startswith("_") and isinstance(getattr(klass, i), str)]
