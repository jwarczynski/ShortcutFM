import numpy as np
from numpy import dtype, ndarray


def get_target_only(
    batch: ndarray[str, dtype[str]],
    sep_token: str,
    cls_token: str | None,
    strip_special_tokens: bool = True,
) -> ndarray[str, dtype[str]]:
    if strip_special_tokens and not cls_token:
        raise ValueError("Cannot strip special tokens if cls_token not provided")

    def decode_and_split(decoded):
        return decoded.split(sep_token)[1]
        # return decoded.split(tokenizer.sep_token, maxsplit=1)[1]

    # Apply function to each element
    vectorized_func = np.vectorize(decode_and_split, otypes=[object])
    targets = vectorized_func(batch)
    if not strip_special_tokens:
        return targets

    # there should be no sep_tokens at this stage
    def strip_cls(decoded):
        return decoded.strip(cls_token)

    vectorized_func = np.vectorize(strip_cls, otypes=[object])
    return vectorized_func(batch)
