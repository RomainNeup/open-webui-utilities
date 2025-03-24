"""
title: Jira Issue
description: This tool allows you to get information about a specific issue on Jira.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
version: 0.2.0
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Split Jira search and Jira get issue
- 0.1.0 - Add support for Personal Access Token authentication and user settings
- 0.2.0 - Add setting for SSL verification
"""


import base64
import json
from typing import Any, Awaitable, Callable, Dict
import requests
from pydantic import BaseModel, Field


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


class Jira:
    def __init__(self, username: str, api_key: str, base_url: str, api_key_auth: bool = True, ssl_verify: bool = True):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key, api_key_auth)
        self.ssl_verify = ssl_verify
        pass

    def get(self, endpoint: str, params: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers, verify=self.ssl_verify)
        if not response.ok:
            raise Exception(f"Failed to get data from Confluence: {response.text}")
        return response.json()

    def get_issue(self, issue_id: str):
        endpoint = f"issue/{issue_id}"
        result = self.get(
            endpoint,
            {"fields": "summary,description,status", "expand": "renderedFields"},
        )

        return {
            "title": result["fields"]["summary"],
            "description": result["renderedFields"]["description"],
            "status": result["fields"]["status"]["name"],
            "link": f"{self.base_url}/browse/{issue_id}",
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

    async def get_issue(
        self,
        issue_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ):
        """
        Get a Jira issue by its ID. The response includes the title, description as HTML, status, and link to the issue.
        :param issue_id: The ID of the issue.
        :return: A response in JSON format (title, description, status, link).
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

        jira = Jira(api_username, api_key, self.valves.base_url, api_key_auth, self.valves.ssl_verify)

        await event_emitter.emit_status(f"Getting issue {issue_id}", False)
        try:
            response = jira.get_issue(issue_id)
            await event_emitter.emit_status(f"Got issue {issue_id}", True)
            await event_emitter.emit_source(
                response["title"], response["link"], response["description"], True
            )
            return json.dumps(response)
        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to get issue {issue_id}: {e}", True, True
            )
            return f"Error: {e}"
