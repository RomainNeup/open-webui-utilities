"""
title: Confluence search
description: This tool allows you to search for and retrieve content from Confluence.
repository: https://github.com/RomainNeup/open-webui-utilities
author: @romainneup
author_url: https://github.com/RomainNeup
funding_url: https://github.com/sponsors/RomainNeup
requirements: markdownify
version: 0.1.2
changelog:
- 0.0.1 - Initial code base.
- 0.0.2 - Fix Valves variables
- 0.1.0 - Split Confuence search and Confluence get page
- 0.1.1 - Split Confluence search by title and by content
- 0.1.2 - Improve search by splitting query into words
"""

import base64
import json
import requests
from typing import Awaitable, Callable, Dict, List, Any
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


class Confluence:
    def __init__(self, username: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.headers = self.authenticate(username, api_key)
        pass

    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers)
        return response.json()

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/{endpoint}"
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def search_by_title(self, query: str) -> List[str]:
        endpoint = "content/search"
        # Split query into individual terms and join them with OR such that each word is optional
        terms = query.split()
        if terms:
            cql_terms = " OR ".join([f'title ~ "{term}"' for term in terms])
        else:
            cql_terms = f'title ~ "{query}"'
        params = {
            'cql': f'({cql_terms}) AND type="page"',
            'limit': 5
        }
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["results"]:
            response.append(item["id"])
        return response

    def search_by_content(self, query: str) -> List[str]:
        endpoint = "content/search"
        # Split query into individual terms and join them with OR such that each word is optional
        terms = query.split()
        if terms:
            cql_terms = " OR ".join([f'text ~ "{term}"' for term in terms])
        else:
            cql_terms = f'text ~ "{query}"'
        params = {
            'cql': f'({cql_terms}) AND type="page"',
            'limit': 5
        }
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["results"]:
            response.append(item["id"])
        return response

    def search_by_title_and_content(self, query: str) -> List[str]:
        endpoint = "content/search"
        # Split query into words and join them with OR; each word is optional.
        terms = query.split()
        if terms:
            cql_terms = " OR ".join([f'title ~ "{term}" OR text ~ "{term}"' for term in terms])
        else:
            cql_terms = f'title ~ "{query}" OR text ~ "{query}"'
        params = {
            'cql': f'({cql_terms}) AND type="page"',
            'limit': 5
        }
        rawResponse = self.get(endpoint, params)
        response = []
        for item in rawResponse["results"]:
            response.append(item["id"])
        return response

    def get_page(self, page_id: str) -> Dict[str, str]:
        endpoint = f"content/{page_id}"
        params = {"expand": "body.view", "include-version": "false"}
        result = self.get(endpoint, params)
        return {
            "id": result["id"],
            "title": result["title"],
            "body": markdownify(result["body"]["view"]["value"]),
            "link": f"{self.base_url}{result['_links']['webui']}",
        }

    def authenticate(self, username: str, api_key: str) -> Dict[str, str]:
        auth_string = f"{username}:{api_key}"
        encoded_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )
        return {"Authorization": "Basic " + encoded_auth_string}


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        pass

    class Valves(BaseModel):
        username: str = Field("example@example.com", description="Your username here")
        api_key: str = Field("ABCD1234", description="Your API key here")
        base_url: str = Field(
            "https://example.atlassian.net/wiki",
            description="The base URL of your Confluence instance",
        )

    # Get content from Confluence
    async def search_confluence(
        self,
        query: str,
        type: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ) -> str:
        """
        Search for a query on Confluence. This returns the result of the search on Confluence.
        Use it to search for a query on Confluence. When a user mentions a search on Confluence, this must be used.
        It can search by content or by title.
        Note: This returns a list of pages that match the search query.
        :param query: The text to search for on Confluence or the title of the page if asked to search by title. MUST be a string.
        :param type: The type of search to perform ('content' or 'title' or 'title_and_content')
        :return: A list of search results from Confluence in JSON format (id, title, body, link). If no results are found, an empty list is returned.
        """
        confluence = Confluence(
            self.valves.username, self.valves.api_key, self.valves.base_url
        )
        event_emitter = EventEmitter(__event_emitter__)

        search_type = type.lower()

        await event_emitter.emit_status(
            f"Searching for {search_type} '{query}' on Confluence...", False
        )
        try:
            if search_type == "title":
                searchResponse = confluence.search_by_title(query)
            elif search_type == "content":
                searchResponse = confluence.search_by_content(query)
            else:
                searchResponse = confluence.search_by_title_and_content(query)
            results = []
            for item in searchResponse:
                result = confluence.get_page(item)
                await event_emitter.emit_source(
                    result["title"], result["link"], result["body"]
                )
                results.append(result)
            await event_emitter.emit_status(
                f"Search for {search_type} '{query}' on Confluence complete. ({len(searchResponse)} results found)",
                True,
            )
            return json.dumps(results)
        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to search for {search_type} '{query}': {e}.", True, True
            )
            return f"Error: {e}"
