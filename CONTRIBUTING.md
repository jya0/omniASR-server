# Contributing to omniASR Streaming Server

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/ARahim3/omniASR-server.git
   cd omniASR-server
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Setup

### Running Locally

```bash
# Start the server
python server.py

# Run tests
python test_streaming.py mic      # Test microphone
python test_streaming.py rest     # Test REST API
python test_streaming.py websocket # Test WebSocket
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Keep functions focused and small
- Add docstrings for public functions

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for new audio format
fix: resolve WebSocket connection timeout
docs: update installation instructions
refactor: simplify audio buffer logic
```

## Pull Request Process

1. Create a branch for your changes
2. Make your changes with clear commits
3. Test your changes locally
4. Update documentation if needed
5. Submit a pull request with:
   - Clear description of changes
   - Any related issue numbers
   - Screenshots/logs if applicable

## Areas for Contribution

### High Priority

- [ ] Benchmarking suite
- [ ] More test coverage
- [ ] Performance optimizations
- [ ] Documentation improvements

### Feature Ideas

- [ ] SSE streaming for REST API
- [ ] Batch processing endpoint
- [ ] Prometheus metrics
- [ ] Model caching improvements
- [ ] Support for more audio formats

### Known Issues

- Streaming quality with short chunks
- Language detection in streaming mode

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
