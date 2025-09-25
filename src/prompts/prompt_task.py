def forcast_prompt(text: str) -> str:
    return f"Make forecast of the value for :\n\n{text} the next 3 months. Describe the trend with 3 bulelts why it will go up or down"