def requires(name: str, flag: bool):
    """
    A decorator that marks a function as requiring an optional dependency.
    If the function is is not available, the function will raise an ImportError.
    """
    def decorator(f):
        if not flag:
            raise ImportError(f"This requires {name}.")
        return f

    return decorator
