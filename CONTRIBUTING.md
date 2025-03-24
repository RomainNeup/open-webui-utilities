# Contributing to Open WebUI Tools

Thank you for your interest in contributing to Open WebUI Tools! This repository provides a collection of web-based utilities for enhancing OpenWebUI, particularly for interacting with Atlassian products like Jira and Confluence.

## Getting Started

1. **Fork the repository**: Start by forking the repository to your own GitHub account.
2. **Clone your fork**: Clone your fork to your local machine.
3. **Set up your environment**: Make sure you have Python 3.8+ installed.

## Development Guidelines

### Coding Standards

1. **Type Annotations**: Use proper type annotations for all function parameters and return types.
2. **Docstrings**: Include detailed docstrings for all functions following the format in the example.
3. **Error Handling**: Implement proper error handling and provide descriptive error messages.
4. **Event Emission**: Use the EventEmitter class to communicate progress and results to the UI.

### Valves (Configuration)

- **Tool Valves**: Define default configurations that the admin sets for all users
- **User Valves**: Define configurations that individual users can customize

### Versioning and Changelog

When modifying an existing tool or creating a new one:

1. **Update the version**: Increment the tool's version number in the docstring header according to semantic versioning:
   - PATCH (0.0.x): For backwards-compatible bug fixes
   - MINOR (0.x.0): For new functionality that's backwards-compatible
   - MAJOR (x.0.0): For incompatible API changes

2. **Add a changelog entry**: For every modification, add a new line to the changelog in the tool's docstring header:
   ```python
   changelog:
   - 0.1.0 - Initial version
   - 0.1.1 - Fixed bug in error handling
   - 0.2.0 - Added new feature X
   ```

## Adding a New Tool

1. Create a new Python file in the appropriate directory:
   - For general tools: `/open-webui-utilities/`
   - For specific integrations: Create a new directory if needed

2. Structure your file according to the template above

3. Test your tool thoroughly before submitting

## Testing

1. Install the tool in a test Open WebUI environment
2. Verify all functionality works as expected
3. Test edge cases and error handling

## Submitting Changes

1. **Create a branch**: Create a branch in your fork for your changes
2. **Commit your changes**: Make small, focused commits with clear messages
3. **Update documentation**: Update any relevant documentation
4. **Submit a pull request**: Create a pull request to the main repository
5. **Describe your changes**: Include a clear description of what your changes do and why they're needed

## Documentation

- Update the README.md file with details about your new tool if applicable
- Include examples of how to use your tool in the documentation

## Versioning

We use semantic versioning:
- Increment the PATCH version for backwards-compatible bug fixes
- Increment the MINOR version for new functionality that's backwards-compatible
- Increment the MAJOR version for incompatible API changes

## License

By contributing to this repository, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have any questions or need help, please open an issue in the repository or contact the project maintainers.
