"""
title: Confluence page
description: This tool allows you to retrieve content from a specific page on Confluence.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
requirements: markdownify
version: 0.2.1
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Fix Valves variables
- 0.1.0 - Split Confuence search and Confluence get page
- 0.1.1 - Add support for Personal Access Token authentication and user settings
- 0.2.0 - Add setting for SSL verification
- 0.2.1 - Add option to strip base64/all images from page content to save context
"""

import base64
import json
import re
import requests
from typing import Awaitable, Callable, Dict, Any
from pydantic import BaseModel, Field
from markdownify import markdownify


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Awaitable[None]]):
        self.event_emitter = event_emitter
        pass

    async def emit_status(self, description: str, done: bool, error: bool = False):
        await self.event_emitter(
            {
                "data": {
                    "description": f"{done and (error and '❌' or '✅') or '🔎'} {description}",
                    "status": done and "complete" or "in_progress",
                    "done": done,
                },
                "type": "status",
            }
        )

    async def emit_message(self, content: str):
        await self.event_emitter({"data": {"content": content}, "type": "message"})

    async def emit_source(self, name: str, url: str, content: str, html: bool = False):
        await self.event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": html}],
                    "source": {"name": name},
                },
            }
        )


def strip_images(text: str, strip_all: bool = False) -> str:
    """Remove image content from markdown text.
    If strip_all is True, removes all markdown images.
    Otherwise, only removes base64 data URI images."""
    if strip_all:
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    else:
        text = re.sub(r"!\[[^\]]*\]\(data:[^)]+\)", "", text)
    # Clean up excessive blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


class Confluence:
    def __init__(
        self, username: str, api_key: str, base_url: str, api_key_auth: bool = True, ssl_verify: bool = True
    ):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key, api_key_auth)
        self.ssl_verify = ssl_verify
        pass

    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers, verify=self.ssl_verify)
        if not response.ok:
            raise Exception(f"Failed to get data from Confluence: {response.text}")
        return response.json()

    def get_page(self, page_id: str, strip_images_mode: str = "none") -> Dict[str, Any]:
        endpoint = f"content/{page_id}"
        params = {"expand": "body.view", "include-version": "false"}
        result = self.get(endpoint, params)
        body = markdownify(result["body"]["view"]["value"])

        # Strip images if configured
        if strip_images_mode == "all":
            body = strip_images(body, strip_all=True)
        elif strip_images_mode == "base64":
            body = strip_images(body, strip_all=False)

        return {
            "id": result["id"],
            "title": result["title"],
            "body": body,
            "link": f"{self.base_url}{result['_links']['webui']}",
        }

    def authenticate_api_key(self, username: str, api_key: str) -> Dict[str, str]:
        auth_string = f"{username}:{api_key}"
        encoded_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )
        return {"Authorization": "Basic " + encoded_auth_string}

    def authenticate_personal_access_token(self, access_token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {access_token}"}

    def authenticate(self, username: str, api_key: str, api_key_auth: bool) -> Dict[str, str]:
        if api_key_auth:
            return self.authenticate_api_key(username, api_key)
        else:
            return self.authenticate_personal_access_token(api_key)


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        pass

    class Valves(BaseModel):
        base_url: str = Field(
            "https://example.atlassian.net/wiki",
            description="The base URL of your Confluence instance",
        )
        username: str = Field(
            "example@example.com",
            description="Default username (leave empty for personal access token)",
        )
        api_key: str = Field(
            "ABCD1234", description="Default API key or personal access token"
        )
        ssl_verify: bool = Field(
            True,
            description="SSL verification"
        )
        strip_images: str = Field(
            default="base64",
            description="Strip images from page content to save context. "
            "'none' = keep all images, 'base64' = strip only embedded base64 images, "
            "'all' = strip all images.",
        )
        pass

    class UserValves(BaseModel):
        api_key_auth: bool = Field(
            True,
            description="Use API key authentication; disable this to use a personal access token instead.",
        )
        username: str = Field(
            "",
            description="Username, typically your email address; leave empty if using a personal access token or default settings.",
        )
        api_key: str = Field(
            "",
            description="API key or personal access token; leave empty to use the default settings.",
        )
        pass

    # Get content from Confluence
    async def get_confluence_page(
        self,
        page_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {}
    ) -> str:
        """
        Get the content of a page on Confluence. This returns the content of a specific page on Confluence.
        Use it to get the content of a specific page on Confluence. When a user requests a specific page, this must be used.
        :param page_id: The ID of the page on Confluence
        :return: The content of the page on Confluence in JSON format (title, body, link). If the page is not found, an error message is returned.
        """
        event_emitter = EventEmitter(__event_emitter__)

        # Get the username and API key
        if __user__ and "valves" in __user__:
            user_valves = __user__["valves"]
            api_key_auth = user_valves.api_key_auth
            api_username = user_valves.username or self.valves.username
            api_key = user_valves.api_key or self.valves.api_key
        else:
            api_username = self.valves.username
            api_key = self.valves.api_key
            api_key_auth = True
        
        if (api_key_auth and not api_username) or not api_key:
            await event_emitter.emit_status(
                "Please provide a username and API key or personal access token.", True, True
            )
            return "Error: Please provide a username and API key or personal access token."
        
        confluence = Confluence(
            api_username, api_key, self.valves.base_url, api_key_auth, self.valves.ssl_verify
        )
        
        await event_emitter.emit_status(f"Retrieving page '{page_id}' from Confluence...", False)
        try:
            result = confluence.get_page(page_id, self.valves.strip_images)
            await event_emitter.emit_status(f"Retrieved page '{page_id}' from Confluence.", True)
            await event_emitter.emit_source(result["title"], result["link"], result["body"])
            return json.dumps(result)
        except Exception as e:
            await event_emitter.emit_status(f"Failed to retrieve page '{page_id}': {e}.", True, True)
            return f"Error: {e}"
