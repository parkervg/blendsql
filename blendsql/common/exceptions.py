class InvalidBlendSQL(ValueError):
    pass


class LMFunctionException(ValueError):
    pass


class TypeResolutionException(LMFunctionException):
    pass
