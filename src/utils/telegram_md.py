import re


MDV2_SPECIALS = r"_*[]()~`>#+-=|{}.!"


def escape_md_v2(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"([\\_\*\[\]\(\)~`>\#\+\-=\|\{\}\.\!])", r"\\\1", str(text))
