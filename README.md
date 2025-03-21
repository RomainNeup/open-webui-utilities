# Open WebUI Tools Repository

## Introduction
This repository provides a collection of web-based tools for interacting with Jira and Confluence APIs. The tools are designed to be easily installed and configured within the Open WebUI workspace.

## Installing Tools
To install a tool, you must have admin privileges in your workspace. Follow these steps:
1. Go to Workspace > Tools > "+" and paste the tool code.
2. Give the tool a name and description.
3. Press Save to create the tool.
4. To configure the tool, go back to Workspace > Tools, select the tool you want to configure, and click the cog icon.
5. Enter your username, API key, and base URL in the provided fields.

## Tool Descriptions

### confluence_page.py
This tool retrieves a Confluence page by its ID.

- Configuration:
    - Username: example@mail.com
    - API Key: 1234ABCD
    - Base URL: https://example.atlassian.net/wiki
- User Settings:
    - api_key_auth: Use API key authentication (true) or personal access token (false)
    - username: Override default username for authentication
    - api_key: Override default API key or personal access token
- Usage: Simply run the script and provide the Confluence page ID as an argument. The tool will return the corresponding page content.

### confluence_search.py
This tool searches for Confluence pages using a given query string.

- Configuration:
    - Username: example@mail.com
    - API Key: 1234ABCD
    - Base URL: https://example.atlassian.net/wiki
    - Result Limit: 5
- User Settings:
    - api_key_auth: Use API key authentication (true) or personal access token (false)
    - username: Override default username for authentication
    - api_key: Override default API key or personal access token
    - split_query: Split the query into individual words for better search results
- Usage: Run the script and provide the search query as an argument. The tool will return the results of the search.

### jira_issue.py
This tool retrieves Jira issues based on ID.

- Configuration:
    - Username: example@mail.com
    - API Key: 1234ABCD
    - Base URL: https://example.atlassian.net
- User Settings:
    - api_key_auth: Use API key authentication (true) or personal access token (false)
    - username: Override default username for authentication
    - api_key: Override default API key or personal access token
- Usage: Run the script and provide the ID as arguments. The tool will return the corresponding Jira issues.

### jira_search.py
This tool searches for Jira issues using a given query string.

- Configuration:
    - Username: example@mail.com
    - API Key: 1234ABCD
    - Base URL: https://example.atlassian.net
    - Result Limit: 5
- User Settings:
    - api_key_auth: Use API key authentication (true) or personal access token (false)
    - username: Override default username for authentication
    - api_key: Override default API key or personal access token
    - split_query: Split the query into individual words for better search results
- Usage: Run the script and provide the search query as an argument. The tool will return the results of the search.

## Contributing
This repository is open-source and welcomes contributions from the community. If you'd like to contribute a new tool or improve an existing one, please fork this repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE. file for more information.