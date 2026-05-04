import aiohttp
import aiofiles
import base64


async def encode_local_image(image_url: str) -> str:
    async with aiofiles.open(image_url, "rb") as image_file:
        data = await image_file.read()
        return base64.b64encode(data).decode("utf-8")


async def encode_remote_image(image_url: str, session: aiohttp.ClientSession) -> str:
    async with session.get(image_url) as response:
        response.raise_for_status()
        data = await response.read()
        return base64.b64encode(data).decode("utf-8")


async def get_base64_string(image_url: str, session: aiohttp.ClientSession) -> str:
    if image_url.startswith(("http://", "https://")):
        base64_image = await encode_remote_image(image_url, session)
    else:
        base64_image = await encode_local_image(image_url)
    return base64_image


async def openai_compatible_image_url(
    image_url: str, session: aiohttp.ClientSession
) -> str:
    if image_url.startswith("data:image"):
        return image_url
    filetype = image_url.split(".")[-1].lower()
    if filetype == "jpg":
        filetype = "jpeg"
    base64_image = await get_base64_string(image_url, session)
    return f"data:image/{filetype};base64,{base64_image}"


async def openai_compatible_audio_url(
    audio_url: str, session: aiohttp.ClientSession
) -> dict:
    filetype = audio_url.split(".")[-1].lower()
    base64_audio = await get_base64_string(audio_url, session)
    return {"data": base64_audio, "format": filetype}
